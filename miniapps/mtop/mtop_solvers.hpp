#ifndef MTOP_SOLVERS_HPP
#define MTOP_SOLVERS_HPP

#include "mfem.hpp"
#include "rand_eigensolver.hpp"
#include "general/forall.hpp"

using namespace mfem;

class IsoElasticyLambdaCoeff: public mfem::Coefficient
{
public:
    IsoElasticyLambdaCoeff(mfem::Coefficient* E_, mfem::Coefficient* nu_):E(E_),nu(nu_)
    {

    }


    virtual double Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        double rez=double(0.0);

        double EE=E->Eval(T,ip);
        double nn=nu->Eval(T,ip);

        rez=EE*nn/(1.0+nn);
        rez=rez/(1.0-2.0*nn);

        return rez;
    }

private:
    mfem::Coefficient* E;
    mfem::Coefficient* nu;
};

class IsoElasticySchearCoeff: public Coefficient
{
public:
    IsoElasticySchearCoeff(mfem::Coefficient* E_, mfem::Coefficient* nu_):E(E_),nu(nu_)
    {}

    virtual double Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        double rez=double(0.0);

        double EE=E->Eval(T,ip);
        double nn=nu->Eval(T,ip);

        rez=EE/(1.0+nn); rez=rez/2.0;

        return rez;
    }

private:
    mfem::Coefficient* E;
    mfem::Coefficient* nu;
};


/// operator A is considered to be symmetric
class LocProductOperator : public Operator
{
private:
    const Operator *A, *B;
    mutable Vector z;

public:
    LocProductOperator(const Operator *A, const Operator *B):
        Operator(A->Height(), B->Width()), A(A), B(B), z(A->Width())
    {
        MFEM_VERIFY(A->Width() == B->Height(),
                    "incompatible Operators: A->Width() = " << A->Width()
                                                            << ", B->Height() = " << B->Height());
    }

    void Mult(const Vector &x, Vector &y) const override
    {

        B->Mult(x, z);
        A->Mult(z, y);
    }

    void MultTranspose(const Vector &x, Vector &y) const override
    {

        A->Mult(x, z);
        B->MultTranspose(z, y);
    }

    void MultB(const Vector &x, Vector &y)
    {
        B->Mult(x, y);
    }

    void MultA(const Vector &x, Vector &y)
    {
        A->Mult(x,y);
    }

    virtual ~LocProductOperator(){

    }
};




class LElasticOperator:public mfem::Operator
{
public:
    LElasticOperator(mfem::ParMesh* mesh_, int vorder=1);

    virtual ~LElasticOperator();

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000);

    /// Solves the forward problem.
    void FSolve();

    /// Forms the tangent matrix
    void AssembleTangent();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Solves the forward problem with the provided rhs.
    void FSolve(mfem::Vector& rhs);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

    /// Clear all displacement BC
    void DelDispBC();

    /// Set the values of the volumetric force.
    void SetVolForce(double fx,double fy, double fz=0.0);

    /// Add surface load
    void AddSurfLoad(int id, double fx,double fy, double fz=0.0)
    {
        mfem::Vector vec; vec.SetSize(pmesh->SpaceDimension());
        vec[0]=fx;
        vec[1]=fy;
        if(pmesh->SpaceDimension()==3){vec[2]=fz;}
        mfem::VectorConstantCoefficient* vc=new mfem::VectorConstantCoefficient(vec);
        if(load_coeff.find(id)!=load_coeff.end()){ delete load_coeff[id];}
        load_coeff[id]=vc;
    }

    /// Add surface load
    void AddSurfLoad(int id, mfem::VectorCoefficient& ff)
    {
        surf_loads[id]=&ff;
    }

    /// Associates coefficient to the volumetric force.
    void SetVolForce(mfem::VectorCoefficient& ff);


    /// Returns the displacements.
    mfem::ParGridFunction& GetDisplacements()
    {
        fdisp.SetFromTrueDofs(sol);
        return fdisp;
    }

    /// Returns the adjoint displacements.
    mfem::ParGridFunction& GetADisplacements()
    {
        adisp.SetFromTrueDofs(adj);
        return adisp;
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(mfem::ParGridFunction& sgf){
        sgf.SetSpace(vfes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(mfem::ParGridFunction& agf){
        agf.SetSpace(vfes); agf.SetFromTrueDofs(adj);}

    /// Sets BC dofs, bilinear form, preconditioner and solver.
    /// Should be called before calling Mult of MultTranspose
    virtual void Assemble();

    /// Forward solve with given RHS. x is the RHS vector. The BC are set to zero.
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

    /// Adjoint solve with given RHS. x is the RHS vector. The BC are set to zero.
    virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

    /// Set material
    void SetMaterial(Coefficient& E_, Coefficient& nu_)
    {
        E=&E_;
        nu=&nu_;

        delete lambda;
        delete mu;
        delete bf;

        lambda=new IsoElasticyLambdaCoeff(E,nu);
        mu=new IsoElasticySchearCoeff(E,nu);

        bf=new ParBilinearForm(vfes);
        bf->SetDiagonalPolicy(DIAG_ONE);
        bf->AddDomainIntegrator(new ElasticityIntegrator(*lambda,*mu));
    }

protected:
    mfem::ParMesh* pmesh;

    //solution true vector
    mutable mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;

    //forward solution
    mfem::ParGridFunction fdisp;
    //adjoint solution
    mfem::ParGridFunction adisp;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    mfem::HypreBoomerAMG *prec; //preconditioner
    mfem::CGSolver *ls;  //linear solver

    //finite element space for linear elasticity
    mfem::ParFiniteElementSpace* vfes;
    //finite element collection for linear elasticity
    mfem::FiniteElementCollection* vfec;

    /// Volumetric force created by the solver.
    mfem::VectorConstantCoefficient* lvforce;
    /// Volumetric force coefficient can point to the
    /// one created by the solver or to external vector
    /// coefficient.
    mfem::VectorCoefficient* volforce;

    //surface loads
    std::map<int,mfem::VectorConstantCoefficient*> load_coeff;
    std::map<int,mfem::VectorCoefficient*> surf_loads;

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

    // holds the displacement contrained DOFs
    mfem::Array<int> ess_tdofv;

    //creates a list with essetial dofs
    //sets the values in the bsol vector
    //the list is written in ess_dofs
    void SetEssTDofs(mfem::Vector& bsol, mfem::Array<int>& ess_dofs);

    mfem::Coefficient* E;
    mfem::Coefficient* nu;

    mfem::Coefficient* lambda;
    mfem::Coefficient* mu;
    mfem::Coefficient* rho; //density

    mfem::ParBilinearForm* bf;
    mfem::HypreParMatrix K;

    mfem::ParLinearForm* lf;
};

class FRElasticSolver:public LElasticOperator
{
public:
    FRElasticSolver(mfem::ParMesh* mesh_, int vorder=1, mfem::real_t freq_=0.0);

    virtual ~FRElasticSolver();

    void SetFreq(mfem::real_t freq_){
        freq=freq_;
    }

    void SetDensity(Coefficient& rho_)
    {
        rho=&rho_;
        delete mf;
        mf=new ParBilinearForm(vfes);
        mf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
        //mf->SetAssemblyLevel(mfem::AssemblyLevel::FULL);
        mf->SetDiagonalPolicy(DIAG_ZERO);
        //mf->SetDiagonalPolicy(DIAG_ONE);
        mf->AddDomainIntegrator(new MassIntegrator(*rho));
    }

    void SetDamping(mfem::real_t alpha_, mfem::real_t beta_)
    {
        alpha=alpha_;
        beta=beta_;
    }

    void SetDamping(Coefficient& damp_)
    {
        damp=&damp_;
        delete cf;
        cf=new ParBilinearForm(vfes);
        cf->AddDomainIntegrator(new MassIntegrator(*damp));
    }

    /// Sets BC dofs, bilinear form, preconditioner and solver.
    /// Should be called before calling Mult of MultTranspose
    virtual void Assemble() override;

    void AssembleSVD();

    const RandomizedSubspaceIteration* GetEigSolver(){
        return ss_solver;
    }





protected:

    mfem::real_t freq;
    mfem::real_t alpha;
    mfem::real_t beta;

    mfem::Coefficient* rho; //density
    mfem::Coefficient* damp; //viscous damping

    mfem::ParBilinearForm* mf; //mass bilinear form
    mfem::OperatorHandle hmf;
    mfem::ParBilinearForm* cf; //damping bilinear form
    mfem::OperatorHandle hcf;

    mfem::ParLinearForm* lrf; //real RHS
    mfem::ParLinearForm* lif; //imag RHS


    int num_svd_modes;
    int num_svd_iter;
    LocProductOperator* pop;
    RandomizedSubspaceIteration* ss_solver; //subspace iteration solver

};

/// Applies the PRESB precondition on a block vector [p',q']'
/// where u=x+i*y is the solution and p+i*q is the input.
/// The complex system has the form
/// (W+i*T)(x+i*y)=(p+i*q)
/// The preconditioner is assumed to be applied to the following
/// linear system of equations
/// |W    T| | x| =|p|
/// |T   -W| |-y|  |q|
/// and has the form
/// |W+2T   T|| x|=|e|
/// |T     -W|| y| |g|
///
/// prec_type=2
/// |-W  T| |-x| =|p|
/// |T   W| | y|  |q|
/// and has the form
/// |-W     T|| x|=|e|
/// | T  W+2T|| y| |g|
class PRESBPrec:public IterativeSolver
{
public:
    PRESBPrec(MPI_Comm comm_,int prec_type_=1):
        IterativeSolver(comm_),prec_type(prec_type_)
    {
        comm=comm_;
        MPI_Comm_rank(comm,&myrank);
        a=1.0;
        b=1.0;
    }

    virtual
        ~PRESBPrec(){}


    /// (a*W+i*b*T)(x+i*y)=(e+i*g)
    /// (a*W-i*b*T)(x+i*y)=(e+i*g)
    void SetOperators(Operator* W_, Operator* T_,
                      real_t a_=1.0, real_t b_=1.0){

        this->width=2*(W_->Width());
        this->height=2*(W_->Width());


        a=a_;
        b=b_;

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = W_->Width();
        block_true_offsets[2] = W_->Width();
        block_true_offsets.PartialSum();

        tv.SetSize(W_->Width());
        diag.SetSize(W_->Width());

        W=W_;
        T=T_;

        HypreParMatrix* wm=dynamic_cast<HypreParMatrix*>(W_);
        HypreParMatrix* tm=dynamic_cast<HypreParMatrix*>(T_);

        if((wm!=nullptr)&&(tm!=nullptr)){

             sumop.reset(new HypreParMatrix(*wm));
             (*sumop)*=a;
             sumop->Add(b,*tm);
             amgp.reset(new HypreBoomerAMG());
             amgp->SetOperator(*sumop);
             ls.reset(new mfem::CGSolver(comm));

             //GMRESSolver* pgmres=new mfem::GMRESSolver(comm);
             //pgmres->SetKDim(200);
             //ls.reset(pgmres);
             //ls->SetOperator(*sOp);
             ls->SetOperator(*sumop);
             ls->SetPreconditioner(*amgp);
             ls->iterative_mode=false;
             ls->SetPrintLevel(-1);
        }

        //set MUMPS
        /*
        {
            HypreParMatrix nm(*wm);
            nm*=a;
            nm.Add(2*b,*tm);

            Array2D<const HypreParMatrix*> am(2,2);
            am(0,0)=&nm;
            am(0,1)=tm;
            am(1,0)=tm;
            am(1,1)=wm;

            Array2D<real_t> cm(2,2);
            cm(0,0)=1.0; cm(0,1)=b;
            cm(1,0)=b; cm(1,1)=-a;

            std::unique_ptr<HypreParMatrix> bm;
            bm.reset(HypreParMatrixFromBlocks(am,&cm));


            mumps.reset(new MUMPSSolver(comm));
            //mumps->SetPrintLevel(2);
            //mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
            mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
            mumps->SetOperator(*bm);
        }*/


    }

    virtual void Mult(const Vector &x, Vector &y) const
    {

        //mumps->Mult(x,y);
        //return;

        if(this->iterative_mode){
            ls->iterative_mode=this->iterative_mode;
            //set y[0]=y[0]+y[1]
            const int N=W->Width();
            const bool use_dev = y.UseDevice();
            real_t* yp= y.ReadWrite(use_dev);
            forall(N, [=] MFEM_HOST_DEVICE (int i)
            {
                yp[i]=yp[i]+yp[N+i];
            });
        }

        ls->SetAbsTol(this->abs_tol);
        ls->SetRelTol(this->rel_tol);
        ls->SetMaxIter(this->max_iter);

        BlockVector bvy(y,block_true_offsets);
        Vector& xx=bvy.GetBlock(0);
        Vector& yy=bvy.GetBlock(1);


        if(prec_type==1)
        {//set the RHS (e-g) in tv for prec_type==1
            const int N=W->Width();
            const bool use_dev = tv.UseDevice()||x.UseDevice();
            const real_t* xp= x.Read(use_dev);
            real_t* tp=tv.ReadWrite(use_dev);
            forall(N, [=] MFEM_HOST_DEVICE (int i)
            {
                tp[i]=xp[i]-xp[N+i];
            });
        }else{// set RHS (g-e) in tv for prec_type==2
            const int N=W->Width();
            const bool use_dev = tv.UseDevice()||x.UseDevice();
            const real_t* xp= x.Read(use_dev);
            real_t* tp=tv.ReadWrite(use_dev);
            forall(N, [=] MFEM_HOST_DEVICE (int i)
            {
                tp[i]=xp[N+i]-xp[i];
            });

        }

        //solve (a*W+b*T)(x+y)=(e-g) (prec_type==1)
        //solve (a*W+b*T)(x+y)=(g-e) (prec_type==2)
        ls->Mult(tv,xx);

        if(prec_type==1)
        {
            //solve (a*W+b*T)y=-g+b*T(x+y) (prec_type==1)
            T->Mult(xx,tv);
            const int N=W->Width();
            const bool use_dev = tv.UseDevice()||x.UseDevice();
            const real_t* xp= x.Read(use_dev);
            real_t* tp=tv.ReadWrite(use_dev);
            forall(N, [=] MFEM_HOST_DEVICE (int i)
            {
                tp[i]=-xp[N+i]+b*tp[i];
            });
        }else{
            //solve (a*W+b*T)y=e+a*W(x+y) (prec_type==2)
            W->Mult(xx,tv);
            const int N=W->Width();
            const bool use_dev = tv.UseDevice()||x.UseDevice();
            const real_t* xp= x.Read(use_dev);
            real_t* tp=tv.ReadWrite(use_dev);
            forall(N, [=] MFEM_HOST_DEVICE (int i)
            {
                tp[i]=+xp[i]+a*tp[i];
            });
        }

        ls->Mult(tv,yy);

        //sum (x+y)-y=x
        xx.Add(-1.0,yy);


        //compare to MUMPS
        /*
        Vector tmv(y);
        mumps->Mult(x,tmv);
        tmv.Add(-1,y);
        real_t rr=mfem::InnerProduct(comm, tmv,tmv);
        std::cout<<"Residual="<<sqrt(rr)<<std::endl;
        */
    }

private:
    MPI_Comm comm;
    int myrank;

    Operator* W;
    Operator* T;
    std::unique_ptr<SumOperator> sOp;
    std::unique_ptr<OperatorJacobiSmoother> jOp;
    std::unique_ptr<HypreBoomerAMG> amgp;
    std::unique_ptr<HypreParMatrix> sumop;
    int prec_type;

    real_t a,b;

    std::unique_ptr<IterativeSolver> ls;

    mfem::Array<int> block_true_offsets;

    Vector diag;
    mutable Vector tv;


    std::unique_ptr<MUMPSSolver> mumps;
    std::unique_ptr<HypreParMatrix> bm;


};

class MSP1Prec:public IterativeSolver
{
public:
    MSP1Prec(MPI_Comm comm_):
        IterativeSolver(comm_)
    {
        comm=comm_;
        MPI_Comm_rank(comm,&myrank);
        a=1.0;
        b=1.0;
        c=1.0;
    }

    virtual
        ~MSP1Prec(){}


    /// (a*W1-b*W2+i*c*T)(x+i*y)=(e+i*g)
    void SetOperators(Operator* W1_, Operator* W2_, Operator* T_,
                      real_t a_=1.0, real_t b_=1.0, real_t c_=1.0){

        W1=W1_;
        W2=W2_;
        T=T_;

        a=a_;
        b=b_;
        c=c_;

        this->width=2*(W1_->Width());
        this->height=2*(W1_->Width());


        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = W1_->Width();
        block_true_offsets[2] = W1_->Width();
        block_true_offsets.PartialSum();

        fv.Update(block_true_offsets);

        presb1.reset(new PRESBPrec(comm,1));
        presb2.reset(new PRESBPrec(comm,2));


        presb1->SetOperators(W1,T,a,c);
        presb2->SetOperators(W2,T,b,c);

        bop1.reset(new BlockOperator(block_true_offsets));
        bop2.reset(new BlockOperator(block_true_offsets));

        bop1->SetBlock(0,0,W1,a);
        bop1->SetBlock(0,1,T,c);
        bop1->SetBlock(1,0,T,c);
        bop1->SetBlock(1,1,W1,-a);

        bop2->SetBlock(0,0,W2,-b);
        bop2->SetBlock(0,1,T,c);
        bop2->SetBlock(1,0,T,c);
        bop2->SetBlock(1,1,W2,b);

        ls1.reset(new FGMRESSolver(comm));
        ls1->SetOperator(*bop1);
        ls1->SetPreconditioner(*presb1);

        ls2.reset(new FGMRESSolver(comm));
        ls2->SetOperator(*bop2);
        ls2->SetPreconditioner(*presb2);

        //MUMPS
        {
            HypreParMatrix* km=dynamic_cast<HypreParMatrix*>(W1_);
            HypreParMatrix* mm=dynamic_cast<HypreParMatrix*>(W2_);
            HypreParMatrix* cm=dynamic_cast<HypreParMatrix*>(T_);

            Array2D<const HypreParMatrix*> am(2,2);
            am(0,0)=km;
            am(0,1)=cm;
            am(1,0)=cm;
            am(1,1)=km;

            Array2D<real_t> cc(2,2);
            cc(0,0)=a; cc(0,1)=-c;
            cc(1,0)=c; cc(1,1)=a;

            std::unique_ptr<HypreParMatrix> bm;
            bm.reset(HypreParMatrixFromBlocks(am,&cc));
            mumps1.reset(new MUMPSSolver(comm));
            mumps1->SetPrintLevel(1);
            mumps1->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
            mumps1->SetOperator(*bm);

            am(0,0)=mm;
            am(0,1)=cm;
            am(1,0)=cm;
            am(1,1)=mm;

            cc(0,0)=b; cc(0,1)=c;
            cc(1,0)=-c; cc(1,1)=b;

            bm.reset(HypreParMatrixFromBlocks(am,&cc));

            mumps2.reset(new MUMPSSolver(comm));
            mumps2->SetPrintLevel(1);
            mumps2->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
            mumps2->SetOperator(*bm);
        }
    }

    virtual void Mult(const Vector &x, Vector &y) const
    {

        xb.Update(const_cast<Vector&>(x), block_true_offsets);
        yb.Update(y, block_true_offsets);

        Vector& ee=fv.GetBlock(0);
        Vector& gg=fv.GetBlock(1);

        ls1->SetAbsTol(this->abs_tol);
        ls1->SetRelTol(this->rel_tol);
        ls1->SetMaxIter(this->max_iter);
        ls1->iterative_mode=this->iterative_mode;

        ls2->SetAbsTol(this->abs_tol);
        ls2->SetRelTol(this->rel_tol);
        ls2->SetMaxIter(this->max_iter);
        ls2->iterative_mode=this->iterative_mode;


        ls1->SetPrintLevel(-1);
        ls2->SetPrintLevel(-1);


        bool cflag=true;
        int iter=0;
        while(cflag)
        {
            //Step 1
            //(a*W1+i*c*T) x = b*W2*x+f
            //form ee+i*gg=b*W2*x+f
            W2->Mult(yb.GetBlock(0),ee);
            W2->Mult(yb.GetBlock(1),gg);
            ee*=b; gg*=b;
            ee.Add(1.0,xb.GetBlock(0));
            gg.Add(1.0,xb.GetBlock(1));

            {
                real_t gp=InnerProduct(comm,fv,fv);
                if(myrank==0){
                    std::cout<<"Step1 it:"<< iter<<" "<<gp<<std::endl;}
            }

            /*
            if(iterative_mode){yb.GetBlock(1)*=-1.0;}
            ls1->Mult(fv,y);
            yb.GetBlock(1)*=-1.0;
            */
            mumps1->Mult(fv,y);


            //Step 2
            //(b*W2-i*c*T) x=a*W1*x-f
            //form ee+i*gg=a*W1*x-f

            W1->Mult(yb.GetBlock(0),ee);
            W1->Mult(yb.GetBlock(1),gg);
            ee*=a; gg*=a;
            ee.Add(-1.0,xb.GetBlock(0));
            gg.Add(-1.0,xb.GetBlock(1));

            {
                real_t gp=InnerProduct(comm,fv,fv);
                if(myrank==0){
                    std::cout<<"Step2 it:"<< iter<<" "<<gp<<std::endl;}
            }

            mumps2->Mult(fv,y);

            /*
            if(iterative_mode){yb.GetBlock(0)*=-1.0;}
            ls2->Mult(fv,y);
            yb.GetBlock(0)*=-1.0;
            */


            iter++;
            if(iter>=this->max_iter){cflag=false;}

        }
        std::cout<<std::endl;
        if(myrank==0){std::cout<<"END MSP1"<<std::endl;}
    }

private:
    MPI_Comm comm;
    int myrank;
    real_t a,b,c;

    Operator* W1;
    Operator* W2;
    Operator* T;

    std::unique_ptr<PRESBPrec> presb1;
    std::unique_ptr<PRESBPrec> presb2;

    std::unique_ptr<BlockOperator> bop1;
    std::unique_ptr<BlockOperator> bop2;

    std::unique_ptr<IterativeSolver> ls1;
    std::unique_ptr<IterativeSolver> ls2;

    mfem::Array<int> block_true_offsets;

    mutable BlockVector fv;
    mutable BlockVector tv;

    mutable BlockVector xb;
    mutable BlockVector yb;
    mutable Vector tmp;


    std::unique_ptr<MUMPSSolver> mumps1;
    std::unique_ptr<HypreParMatrix> bm1;
    std::unique_ptr<MUMPSSolver> mumps2;
    std::unique_ptr<HypreParMatrix> bm2;

};

class MSP3Prec:public IterativeSolver
{
public:
    MSP3Prec(MPI_Comm comm_):
        IterativeSolver(comm_)
    {
        comm=comm_;
        MPI_Comm_rank(comm,&myrank);
        a=1.0;
        b=1.0;
        c=1.0;
        alpha=1.0;
    }

    virtual
        ~MSP3Prec(){}


    /// (a*W1-b*W2+i*c*T)(x+i*y)=(e+i*g)
    void SetOperators(Operator* W1_, Operator* W2_, Operator* T_,
                      real_t a_=1.0, real_t b_=1.0, real_t c_=1.0, real_t alpha_=1.0){

        W1=W1_;
        W2=W2_;
        T=T_;

        a=a_;
        b=b_;
        c=c_;

        alpha=alpha_;

        this->width=2*(W1_->Width());
        this->height=2*(W1_->Width());

        tv.SetSize(W1_->Width());

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = W1_->Width();
        block_true_offsets[2] = W1_->Width();
        block_true_offsets.PartialSum();

        fv.Update(block_true_offsets);

        //MUMPS
        {
            HypreParMatrix* km=dynamic_cast<HypreParMatrix*>(W1_);
            HypreParMatrix* mm=dynamic_cast<HypreParMatrix*>(W2_);
            HypreParMatrix* cm=dynamic_cast<HypreParMatrix*>(T_);

            Array2D<const HypreParMatrix*> am(2,2);
            am(0,0)=cm;
            am(0,1)=mm;
            am(1,0)=mm;
            am(1,1)=cm;

            Array2D<real_t> cc(2,2);
            cc(0,0)=c*alpha; cc(0,1)=-b;
            cc(1,0)=b;       cc(1,1)=c*alpha;

            std::unique_ptr<HypreParMatrix> bm;
            bm.reset(HypreParMatrixFromBlocks(am,&cc));
            mumps1.reset(new MUMPSSolver(comm));
            mumps1->SetPrintLevel(1);
            mumps1->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
            mumps1->SetOperator(*bm);

            am(0,0)=cm;
            am(0,1)=km;
            am(1,0)=km;
            am(1,1)=cm;

            cc(0,0)=c*alpha; cc(0,1)=a;
            cc(1,0)=-a;      cc(1,1)=c*alpha;

            bm.reset(HypreParMatrixFromBlocks(am,&cc));

            mumps2.reset(new MUMPSSolver(comm));
            mumps2->SetPrintLevel(1);
            mumps2->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
            mumps2->SetOperator(*bm);

        }

    }


    virtual void Mult(const Vector &x, Vector &y) const
    {
        xb.Update(const_cast<Vector&>(x), block_true_offsets);
        yb.Update(y, block_true_offsets);


        Vector& ee=fv.GetBlock(0);
        Vector& gg=fv.GetBlock(1);

        /*

        ls1->SetAbsTol(this->abs_tol);
        ls1->SetRelTol(this->rel_tol);
        ls1->SetMaxIter(this->max_iter);
        ls1->iterative_mode=this->iterative_mode;

        ls2->SetAbsTol(this->abs_tol);
        ls2->SetRelTol(this->rel_tol);
        ls2->SetMaxIter(this->max_iter);
        ls2->iterative_mode=this->iterative_mode;

        ls1->SetPrintLevel(-1);
        ls2->SetPrintLevel(-1);

        */


        bool cflag=true;
        int iter=0;
        while(cflag)
        {
            //Step 1
            //set RHS
            ee.Set(+1.0,xb.GetBlock(1));
            gg.Set(-1.0,xb.GetBlock(0));

            T->Mult(yb.GetBlock(0),tv); ee.Add((alpha-1.0)*c,tv);
            T->Mult(yb.GetBlock(1),tv); gg.Add((alpha-1.0)*c,tv);
            W1->Mult(yb.GetBlock(1),tv); ee.Add(-a,tv);
            W1->Mult(yb.GetBlock(0),tv); ee.Add(+a,tv);


            /*
            {
                real_t gp=InnerProduct(comm,fv,fv);
                if(myrank==0){
                    std::cout<<"Step1 it:"<< iter<<" "<<gp<<std::endl;}
            }*/


            mumps1->Mult(fv,y);


            //Step 2
            //set RHS
            ee.Set(+1.0,xb.GetBlock(1));
            gg.Set(-1.0,xb.GetBlock(0));
            T->Mult(yb.GetBlock(0),tv); ee.Add((alpha-1.0)*c,tv);
            T->Mult(yb.GetBlock(1),tv); gg.Add((alpha-1.0)*c,tv);
            W2->Mult(yb.GetBlock(1),tv); ee.Add(+b,tv);
            W2->Mult(yb.GetBlock(0),tv); ee.Add(-b,tv);

            /*
            {
                real_t gp=InnerProduct(comm,fv,fv);
                if(myrank==0){
                    std::cout<<"Step2 it:"<< iter<<" "<<gp<<std::endl;}
            }*/


            mumps2->Mult(fv,y);

            iter++;
            if(iter>=this->max_iter){cflag=false;}
        }
        std::cout<<std::endl;
        if(myrank==0){std::cout<<"END MSP3"<<std::endl;}


    }

private:
    MPI_Comm comm;
    int myrank;
    real_t a,b,c;
    real_t alpha;

    Operator* W1;
    Operator* W2;
    Operator* T;

    std::unique_ptr<PRESBPrec> presb1;
    std::unique_ptr<PRESBPrec> presb2;

    std::unique_ptr<BlockOperator> bop1;
    std::unique_ptr<BlockOperator> bop2;

    std::unique_ptr<IterativeSolver> ls1;
    std::unique_ptr<IterativeSolver> ls2;

    mfem::Array<int> block_true_offsets;

    mutable BlockVector fv;
    mutable BlockVector xb;
    mutable BlockVector yb;
    mutable Vector tv;

    std::unique_ptr<MUMPSSolver> mumps1;
    std::unique_ptr<MUMPSSolver> mumps2;

};

#endif // MTOP_SOLVERS_HPP
