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

    //zero pressure BC
    void SetZeroMeanPressure(bool val_)
    {
        zero_mean_pres=val_;
    }

    /// Set the Velocity BC on a given ParGridFunction.
    void SetEssVBC(real_t t, ParGridFunction& pgf);


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
   bool zero_mean_pres;

   // holds the velocity constrained DOFs
   mfem::Array<int> ess_tdofv;

   // Volume force coefficient
   std::shared_ptr<VectorCoefficient> vol_force;

   Array<int> block_true_offsets;
   int siz_u;
   int siz_p;

   ConstantCoefficient onecoeff;
   ConstantCoefficient zerocoef;

   /// Extracts the true boundary doffs of the velocity
   void SetEssTDofsV(mfem::Array<int>& ess_dofs);


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



template <int dim = 2>
struct StokesMomentumQFunction
{
   StokesMomentumQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      // velocity gradient in reference space
      const future::tensor<future::dual<double,double>, dim, dim> &dudxi,
      const double &p,
      const future::tensor<double, dim, dim> &J,
      const double &w) const
   {
      constexpr real_t kinematic_viscosity = 1.0;
      auto I = mfem::future::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto viscous_stress = -p * I + 2.0 * kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::future::tuple{-viscous_stress * JxW};
   }
};

template <int dim = 2>
struct StokesMassConservationQFunction
{
   StokesMassConservationQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      // velocity gradient in reference space
      const future::tensor<double, dim, dim> &dudxi,
      const future::tensor<double, dim, dim> &J,
      const double &w) const
   {
      return mfem::future::tuple{tr(dudxi * inv(J)) * det(J) * w};
   }
};


class StokesSolver:public Operator
{
public:
    StokesSolver(ParMesh* mesh, int order_,bool partial_assembly_=false, bool verbose_=false);

    virtual
    ~StokesSolver();

    /// Set the Linear Solver
     void SetLinearSolver(const real_t rtol = 1e-8,
                          const real_t atol = 1e-12,
                          const int miter = 4000){
         linear_atol=atol;
         linear_rtol=rtol;
         linear_iter=miter;
     }

     /// Sets BC dofs, bilinear form, preconditioner and solver.
     /// Should be called before calling Mult of MultTranspose
     virtual void Assemble();


     /// Solves the forward problem.
     void FSolve();

     /// Solves the adjoint with the provided rhs.
     void ASolve(mfem::Vector &rhs);

     /// Solves the forward problem with the provided rhs.
     void FSolve(mfem::Vector &rhs);

     /// Clear all  BC
     void DeleteBC();


     /// Set the values of the volumetric force.
     void SetVolForce(real_t fx, real_t fy, real_t fz = 0.0);

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
     void SetEssVBC(ParGridFunction& pgf);

     /// Set the Pressure BC on a given ParGridFunction.
     void SetEssPBC(ParGridFunction& pgf);

     /// Forward solve with given RHS. x is the RHS vector.
     /// The BC are set to the specified BC.
     void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

     /// Adjoint solve with given RHS. x is the RHS vector.
     /// The BC are set to zero.
     void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

     /// Return velocity grid function
     mfem::ParGridFunction& GetVelocity()
     {
         vel.SetFromTrueDofs(sol.GetBlock(0));
         return vel;
     }

     /// Return pressure grid function
     mfem::GridFunction& GetPressure()
     {
         pre.SetFromTrueDofs(sol.GetBlock(1));
         return pre;
     }

     /// Return velocty FES
     ParFiniteElementSpace* GetVelocitySpace(){return vfes;}

     /// Return pressure FES
     ParFiniteElementSpace* GetPressureSpace(){return pfes;}

     /// Return the block offset for providing RHS vectors for
     /// Mult and MultTranspose.
     Array<int>& GetTrueBlockOffsets(){return block_true_offsets;}


private:
    bool partial_assembly;
    bool verbose;


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

    void SetEssTDofsV(Vector& v);
    void SetEssTDofsP(Vector& p);

    ParGridFunction vel; //velocity
    ParGridFunction pre; //pressure

    std::unique_ptr<HypreParMatrix> A11;
    std::unique_ptr<HypreParMatrix> A12;
    std::unique_ptr<HypreParMatrix> A21;
    std::unique_ptr<HypreParMatrix> A22;

    std::unique_ptr<HypreParMatrix> A11e;
    std::unique_ptr<HypreParMatrix> A12e;
    std::unique_ptr<HypreParMatrix> A21e;
    std::unique_ptr<HypreParMatrix> A22e;

    std::unique_ptr<mfem::ParBilinearForm> bf11;
    std::unique_ptr<mfem::ParBilinearForm> bf22;
    std::unique_ptr<mfem::ParMixedBilinearForm> bf12;
    std::unique_ptr<mfem::ParMixedBilinearForm> bf21;

    std::unique_ptr<BlockOperator> bop;
    BlockVector rhs;
    BlockVector sol;


};


class SpaceOrthOperator:public Operator
{
public:

    SpaceOrthOperator():Operator(-1)
    {
#ifdef MFEM_USE_MPI
        comm=MPI_COMM_SELF;
#endif
    }

    SpaceOrthOperator(MPI_Comm comm_):Operator(-1),comm(comm_)
    {

    }

    /// Adds vector to the vector space.
    /// The first vector determines the size of the operator.
    void AddVector(Vector& v)
    {
        if(0==vspace.size())
        {
            Operator::width=v.Size();
            Operator::height=v.Size();
        }
        vspace.push_back(Vector(v));
    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        y.SetSize(x.Size());
        for(int i=0;i<x.Size();i++){
            y[i]=x[i];
        }
#ifdef MFEM_USE_MPI
        for(int i=0;i<vspace.size();i++){
            real_t r=InnerProduct(comm,y,vspace[i]);
            y.Add(-r,vspace[i]);
        }
#else
        for(int i=0;i<vspace.size();i++){
            real_t r=InnerProduct(y,vspace[i]);
            y.Add(-r,vspace[i]);
        }

#endif
    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {
        Mult(x,y);
    }

private:
    std::vector<Vector> vspace;
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif

};


///The LSC preconditioner based on:
///H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth, and R. Tuminaro,
/// “Block Preconditioners Based on Approximate Commutators,”
/// SIAM Journal on Scientific Computing, vol. 27, pp. 1651–1668, 2006.
/// The preconditioner is applied to the following system
/// [F  G] |u| = |f|
/// [D -C] |p| = |g|
/// and has the form
/// [Fa^{-1}  0    ]
/// [0      Sa^{-1}]
/// where Fa approximates the inverse of F
/// and Sa approximates the Schur complement DF^{-1}G
/// Sa^{-1}=(D S^{-1} G)^{-1} (D S^{-1} F S^{-1} G) (D S^{-1} G)^{-1}
/// S is optional diagonal scaling matrix which can be taken to be
/// S=diag(G) or S=lump(G) where G is a mass matrix in the discrete space of u.
class DLSCPrec:public IterativeSolver
{
public:

    DLSCPrec(HypreParMatrix* F_, HypreParMatrix* D_, HypreParMatrix* G_,
            bool zero_mean_pres_,
            Vector* S_=nullptr, int dim=-1,
             int print_lvl=-1):
        zero_mean_pres(zero_mean_pres_),F(F_), D(D_), G(G_), S(S_)
    {
        //form A=D*S^{-1}*G if S!=nullptr
        //or A=D*G if S==nullptr
        if(nullptr==S){
            A.reset(ParMult(D,G));
        }else{
            HypreParMatrix T(*G);
            T.InvScaleRows(*S);
            A.reset(ParMult(D,&T));
        }

        //set the preconditioner for F
        {
            HypreBoomerAMG* p=new HypreBoomerAMG();
            p->SetPrintLevel(print_lvl);
            p->SetOperator(*F);
            if((-1)!=dim){p->SetSystemsOptions(dim, true);}
            pr11.reset(p);

        }

        //set the solver and the preconditioner for A
        {
            HypreBoomerAMG* p=new HypreBoomerAMG();
            p->SetPrintLevel(print_lvl);
            p->SetOperator(*A);
            if(zero_mean_pres){
                std::cout<<"Ortho Solver"<<std::endl;
                OrthoSolver* os=new OrthoSolver(F->GetComm());
                os->SetSolver(*p);
                fpaa.reset(os);
                lsaa.reset(new GMRESSolver(F->GetComm()));
            }else{
                praa.reset(p);
                fpaa=praa;
                lsaa.reset(new CGSolver(F->GetComm()));
            }

            //lsaa.reset(new GMRESSolver(F->GetComm()));
            //lsaa.reset(new CGSolver(F->GetComm()));
            lsaa->SetOperator(*A);
            lsaa->SetPreconditioner(*fpaa);
            lsaa->SetAbsTol(1e-12);
            lsaa->SetRelTol(1e-12);
            lsaa->SetMaxIter(10);
            lsaa->iterative_mode=true;
        }

        siz_u=F->GetNumRows();
        siz_p=D->GetNumRows();

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = siz_u;
        block_true_offsets[2] = siz_p;
        block_true_offsets.PartialSum();

        tv.Update(block_true_offsets); tv=0.0;
        av.Update(block_true_offsets); av=0.0;
    }

    virtual
    ~DLSCPrec()
    {

    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        xb.Update(const_cast<Vector&>(x), block_true_offsets);
        yb.Update(y, block_true_offsets);

        //u=F^{-1}*f
        pr11->Mult(xb.GetBlock(0),yb.GetBlock(0));

        //Schur complement
        lsaa->Mult(xb.GetBlock(1),tv.GetBlock(1));
        G->Mult(tv.GetBlock(1),tv.GetBlock(0));
        if(nullptr!=S){
            for(int i=0;i<S->Size();i++){
                tv.GetBlock(0)[i]=(tv.GetBlock(0))[i]/(*S)[i];
            }
        }
        F->Mult(tv.GetBlock(0),av.GetBlock(0));
        if(nullptr!=S){
            for(int i=0;i<S->Size();i++){
                av.GetBlock(0)[i]=(av.GetBlock(0))[i]/(*S)[i];
            }
        }
        D->Mult(av.GetBlock(0),av.GetBlock(1));

        /*
        real_t dd=InnerProduct(F->GetComm(),av.GetBlock(1),av.GetBlock(1));
        std::cout<<"nr="<<dd<<std::endl;
        dd=InnerProduct(F->GetComm(),yb.GetBlock(1),yb.GetBlock(1));
        std::cout<<"br="<<dd<<std::endl;
        */

        lsaa->Mult(av.GetBlock(1),yb.GetBlock(1));
    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {
        Mult(x,y);
    }

    virtual void SetPrintLevel(int print_lvl) override
    {
        lsaa->SetPrintLevel(print_lvl);
    }

    virtual void SetAbsTol(real_t tol_)
    {
        lsaa->SetAbsTol(tol_);
    }

    virtual void SetRelTol(real_t tol_)
    {
       lsaa->SetRelTol(tol_);
    }

    virtual void SetMaxIter(int it)
    {
        lsaa->SetMaxIter(it);
    }


private:
    bool zero_mean_pres;

    HypreParMatrix* F;
    HypreParMatrix* D;
    HypreParMatrix* G;
    Vector* S;

    std::shared_ptr<HypreParMatrix> A;

    //the preconditioner used for solving A^{-1}*RHS
    //holds either praa or OrthoSolver(praa)
    //depending on the pressure boundary conditions
    std::shared_ptr<Solver> fpaa;
    std::shared_ptr<Solver> praa; //AMG preconditioner for A
    std::unique_ptr<IterativeSolver> lsaa; //solver approximating A^{-1}
    std::unique_ptr<Solver> pr11; //preconditioner for F

    int siz_u;
    int siz_p;
    Array<int> block_true_offsets;
    mutable BlockVector tv;
    mutable BlockVector av;
    mutable BlockVector yb;
    mutable BlockVector xb;

};



/// Block diagonal preconditioner for time dependent
/// Stokes problem
/// K.-A. Mardal and R. Winther, Uniform preconditioners
/// for the time dependent Stokes problem,
/// Numerische Mathematik, vol. 98, Art. no. 2, 2004.
class BlockDiagTSPrec:public IterativeSolver
{
public:
    BlockDiagTSPrec(real_t dt_, std::shared_ptr<Coefficient> visc_,
                    ParFiniteElementSpace* vfes_, ParFiniteElementSpace* pfes_,
                    mfem::Array<int>& ess_tdofv_, mfem::Array<int>& ess_tdofp_):
                        dt(dt_),
                        visc(visc_),
                        vfes(vfes_),
                        pfes(pfes_),
                        ess_tdofv(ess_tdofv_),
                        ess_tdofp(ess_tdofp_)
    {

        orto.reset(new SpaceOrthOperator(pfes->GetComm()));
        {
            ParGridFunction gfone(pfes);
            ConstantCoefficient one(1.0);
            gfone.ProjectCoefficient(one);
            gfone.SetTrueVector();
            orto->AddVector(gfone.GetTrueVector());
        }

        std::unique_ptr<ParBilinearForm> b11(new ParBilinearForm(vfes));
        std::unique_ptr<ParBilinearForm> b22(new ParBilinearForm(pfes));
        std::unique_ptr<ParBilinearForm> bmm(new ParBilinearForm(pfes));

        ConstantCoefficient onecoeff(1.0);
        ConstantCoefficient zerocoef(0.0);

        ProductCoefficient svisc(dt,*visc);
        //RatioCoefficient svisc(1.0,*visc);


        ConstantCoefficient onecoeffdt(1.0/dt);
        b11->AddDomainIntegrator(new  VectorMassIntegrator(onecoeff));
        b11->AddDomainIntegrator(new  ElasticityIntegrator(zerocoef,svisc));
        b11->Assemble(0);
        b11->Finalize(0);
        A11.reset(b11->ParallelAssemble());

        HypreParMatrix* Ae=A11->EliminateRowsCols(ess_tdofv);
        delete Ae;

        int dim=vfes->GetVDim();
        prec11.reset(new HypreBoomerAMG());
        prec11->SetOperator(*A11);
        prec11->SetSystemsOptions(dim, true);

        b22->AddDomainIntegrator(new DiffusionIntegrator());
        b22->Assemble(0);
        b22->Finalize(0);
        A22.reset(b22->ParallelAssemble());
        Ae=A22->EliminateRowsCols(ess_tdofp);
        delete Ae;

        pop.reset(new ProductOperator(orto.get(),A22.get(),false,false));

        prec22.reset(new HypreBoomerAMG());
        prec22->SetOperator(*A22);

        //bmm->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator(svisc)));
        bmm->AddDomainIntegrator(new MassIntegrator(svisc));
        bmm->Assemble(0);
        bmm->Finalize(0);
        M22.reset(bmm->ParallelAssemble());
        Ae=M22->EliminateRowsCols(ess_tdofp);
        delete Ae;

        precmm.reset(new HypreBoomerAMG());
        precmm->SetOperator(*M22);
        lm22.reset(new CGSolver(pfes->GetComm()));
        lm22->SetOperator(*M22);
        lm22->SetPreconditioner(*precmm);
        lm22->SetAbsTol(1e-12);
        lm22->SetRelTol(1e-12);
        lm22->SetMaxIter(10);




        ls11.reset(new CGSolver(vfes->GetComm()));
        ls11->SetOperator(*A11);
        ls11->SetPreconditioner(*prec11);
        ls11->SetAbsTol(1e-12);
        ls11->SetRelTol(1e-12);
        ls11->SetMaxIter(10);

        //ls22.reset(new CGSolver(pfes->GetComm()));
        ls22.reset(new GMRESSolver(pfes->GetComm()));
        //ls22->SetOperator(*pop);
        ls22->SetOperator(*A22);
        orts.reset(new OrthoSolver(pfes->GetComm()));
        orts->SetSolver(*prec22);
        ls22->SetPreconditioner(*orts);
        ls22->SetAbsTol(1e-12);
        ls22->SetRelTol(1e-12);
        ls22->SetMaxIter(10);

        siz_u=vfes->TrueVSize();
        siz_p=pfes->TrueVSize();

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = siz_u;
        block_true_offsets[2] = siz_p;
        block_true_offsets.PartialSum();

        tv.Update(block_true_offsets); tv=0.0;

    }

    virtual ~BlockDiagTSPrec()
    {

    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        xb.Update(const_cast<Vector&>(x), block_true_offsets);
        yb.Update(y, block_true_offsets);

        ls11->Mult(xb.GetBlock(0),yb.GetBlock(0));
        if(0==pfes->GetMyRank())
        {
            std::cout<<"A11 is ready!"<<std::endl;
        }
        ls22->Mult(xb.GetBlock(1),tv.GetBlock(1));
        lm22->Mult(xb.GetBlock(1),yb.GetBlock(1));
        //M22->Mult(xb.GetBlock(1),yb.GetBlock(1));
        yb.GetBlock(1).Add(1.0,tv.GetBlock(1));
        if(0==pfes->GetMyRank())
        {
            std::cout<<"A22 is ready!"<<std::endl;
        }
    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {
        Mult(x,y);
    }

    virtual void SetPrintLevel(int print_lvl) override
    {
        prec11->SetPrintLevel(print_lvl);
        prec22->SetPrintLevel(print_lvl);
        ls11->SetPrintLevel(print_lvl);
        ls22->SetPrintLevel(print_lvl);
        lm22->SetPrintLevel(print_lvl);
    }

    virtual void SetAbsTol(real_t tol_)
    {
        ls11->SetAbsTol(tol_);
        ls22->SetAbsTol(tol_);
        lm22->SetAbsTol(tol_);
    }

    virtual void SetRelTol(real_t tol_)
    {
       ls11->SetRelTol(tol_);
       ls22->SetRelTol(tol_);
       lm22->SetRelTol(tol_);
    }

    virtual void SetMaxIter(int it)
    {
        ls11->SetMaxIter(it);
        ls22->SetMaxIter(it);
        lm22->SetMaxIter(it);
    }

private:
    real_t dt;
    std::shared_ptr<Coefficient> visc;
    ParFiniteElementSpace* vfes;
    ParFiniteElementSpace* pfes;
    mfem::Array<int>& ess_tdofv;
    mfem::Array<int>& ess_tdofp;

    std::unique_ptr<HypreParMatrix> A11;
    std::unique_ptr<HypreParMatrix> A22;
    std::unique_ptr<HypreParMatrix> M22;

    std::unique_ptr<mfem::HypreBoomerAMG> prec11;
    std::unique_ptr<mfem::HypreBoomerAMG> prec22;
    std::unique_ptr<mfem::HypreBoomerAMG> precmm;
    std::unique_ptr<mfem::CGSolver> ls11;
    std::unique_ptr<mfem::IterativeSolver> ls22;
    std::unique_ptr<mfem::CGSolver> lm22;

    int siz_u;
    int siz_p;
    Array<int> block_true_offsets;
    mutable BlockVector tv;
    mutable BlockVector yb;
    mutable BlockVector xb;

    std::unique_ptr<OrthoSolver> orts;
    std::unique_ptr<SpaceOrthOperator> orto;
    std::unique_ptr<ProductOperator> pop;

};



}



#endif // NS_OPERATORS_HPP
