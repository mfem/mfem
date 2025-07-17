#ifndef MTOP_SOLVERS_HPP
#define MTOP_SOLVERS_HPP

#include "mfem.hpp"
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


class IsoLinElasticSolver:public Operator
{
public:
    IsoLinElasticSolver(mfem::ParMesh* mesh_, int vorder=1);

    ~IsoLinElasticSolver();

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
        bf->AddDomainIntegrator(new ElasticityIntegrator(*lambda,*mu));
    }

private:
    mfem::ParMesh* pmesh;

    //solution true vector
    mutable mfem::Vector sol;
    //adjoint true vector
    mutable mfem::Vector adj;
    //RHS
    mutable mfem::Vector rhs;

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
    std::map<int,mfem::VectorCoefficient*> load_coeff; //internaly generated load
    std::map<int,mfem::VectorCoefficient*> surf_loads; //external vector coeeficients

    class SurfaceLoad;
    std::unique_ptr<SurfaceLoad> lcsurf_load; //localy generated surface loads
    std::unique_ptr<SurfaceLoad> glsurf_load; //global surface loads

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
    std::unique_ptr<mfem::HypreParMatrix> K;
    std::unique_ptr<mfem::HypreParMatrix> Ke;

    mfem::ParLinearForm* lf;


    class SurfaceLoad:public VectorCoefficient
    {
    public:
        SurfaceLoad(int dim, std::map<int,VectorCoefficient*>& cmap):VectorCoefficient(dim)
        {
            map=&cmap;
        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                          const IntegrationPoint &ip)
        {
            V.SetSize(GetVDim()); V=0.0;
            auto it=map->find(T.Attribute);
            if(it!=map->end())
            {
                it->second->Eval(V,T,ip);
            }
        }
    private:
        std::map<int,VectorCoefficient*>* map;
    };

};




namespace PointwiseTrans
{

/*  Standrd "Heaviside" projection in topology optimization with threshold eta
 * and steepness of the projection beta.
 * */
inline
    double HProject(double rho, double eta, double beta)
{
    // tanh projection - Wang&Lazarov&Sigmund2011
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double c=std::tanh(beta*(rho-eta));
    double rez=(a+c)/(a+b);
    return rez;
}

/// Gradient of the "Heaviside" projection with respect to rho.
inline
    double HGrad(double rho, double eta, double beta)

{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=beta*(1.0-c*c)/(a+b);
    return rez;
}

/// Second derivative of the "Heaviside" projection with respect to rho.
inline
    double HHess(double rho,double eta, double beta)
{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=-2.0*beta*beta*c*(1.0-c*c)/(a+b);
    return rez;
}


inline
    double FluidInterpolation(double rho,double q)
{
    return q*(1.0-rho)/(q+rho);
}

inline
    double GradFluidInterpolation(double rho, double q)
{
    double tt=q+rho;
    return -q/tt-q*(1.0-rho)/(tt*tt);
}

inline
    double SIMPInterpolation(double rho, double p)
{
    return std::pow(rho,p);
}

inline
    double GradSIMPInterpolation(double rho, double p)
{
    return p*std::pow(rho,p-1.0);
}

}


class FilterOperator: public Operator
{
public:
    FilterOperator(double r_, mfem::ParMesh* pmesh_, int order_=2)
    {
        r=r_;
        order=order_;
        pmesh=pmesh_;
        int dim=pmesh->Dimension();
        sfec=new mfem::H1_FECollection(order, dim);
        sfes=new mfem::ParFiniteElementSpace(pmesh,sfec,1);

        ifec=new mfem::H1Pos_FECollection(order-1,dim);
        ifes=new mfem::ParFiniteElementSpace(pmesh,ifec,1);

        dfes=ifes;
        SetSolver();


        K=nullptr;
        S=nullptr;
        A=nullptr;
        pcg=nullptr;
        prec=nullptr;

        mfem::Operator::width=dfes->GetTrueVSize();
        mfem::Operator::height=sfes->GetTrueVSize();

    }

    FilterOperator(double r_, mfem::ParMesh* pmesh_, mfem::ParFiniteElementSpace* dfes_,int order_=2):dfes(dfes_)
    {
        r=r_;
        order=order_;
        pmesh=pmesh_;
        int dim=pmesh->Dimension();
        sfec=new mfem::H1_FECollection(order, dim);
        sfes=new mfem::ParFiniteElementSpace(pmesh,sfec,1);

        ifec=nullptr;
        ifes=nullptr;

        SetSolver();


        K=nullptr;
        S=nullptr;
        A=nullptr;
        pcg=nullptr;
        prec=nullptr;

        mfem::Operator::width=dfes->GetTrueVSize();
        mfem::Operator::height=sfes->GetTrueVSize();

    }

    mfem::ParFiniteElementSpace* GetFilterFES(){return sfes;}
    mfem::ParFiniteElementSpace* GetDesignFES(){return dfes;}


    virtual
        ~FilterOperator()
    {
        delete pcg;
        delete prec;
        delete K;
        delete S;
        delete A;
        delete sfes;
        delete sfec;
        delete ifes;
        delete ifec;
    }

    void Update()
    {
        sfes->Update();
        dfes->Update();
        Assemble();
    }

    virtual
    void Mult(const Vector &x, Vector &y) const override
    {

        //y=bdrc;
        tmpv.SetSize(y.Size());

        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);
        S->Mult(x,tmpv);
        K->EliminateBC(*A,ess_tdofv,bdrc,tmpv);
        pcg->Mult(tmpv,y);

    }

    virtual
    void MultTranspose(const Vector &x, Vector &y) const override
    {
        y=0.0;
        rhsv.SetSize(x.Size()); rhsv=x;
        tmpv.SetSize(x.Size()); tmpv=0.0;
        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);
        K->EliminateBC(*A,ess_tdofv,tmpv,rhsv);
        pcg->Mult(rhsv,tmpv);
        S->MultTranspose(tmpv,y);
    }

    void SetSolver(double rtol_=1e-8, double atol_=1e-12,int miter_=1000, int prt_level_=1)
    {
        rtol=rtol_;
        atol=atol_;
        max_iter=miter_;
        prt_level=prt_level_;
    }

    void AddBC(int id, double val)
    {
        bcr[id]=mfem::ConstantCoefficient(val);

        delete pcg;
        delete prec;
        delete K;
        delete S;
        delete A;

        pcg=nullptr;
        prec=nullptr;
        K=nullptr;
        S=nullptr;
        A=nullptr;

    }

    void Assemble()
    {
        delete pcg;
        delete prec;
        delete K;
        delete S;
        delete A;

        ess_tdofv.DeleteAll();
        bdrc.SetSize(sfes->GetTrueVSize()); bdrc=0.0;
        //set boundary conditions
        if(bcr.size()!=0)
        {
            mfem::ParGridFunction tmpgf(sfes); tmpgf=0.0;
            for(auto it=bcr.begin();it!=bcr.end();it++)
            {
                mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
                ess_bdr=0;
                ess_bdr[it->first -1]=1;
                mfem::Array<int> ess_tdof_list;
                sfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
                ess_tdofv.Append(ess_tdof_list);
                tmpgf.ProjectBdrCoefficient(it->second,ess_bdr);
            }
            tmpgf.GetTrueDofs(bdrc);
        }


        double dr=r/(2.0*sqrt(3.0));
        mfem::ConstantCoefficient dc(dr*dr);

        mfem::ParBilinearForm* bf=new mfem::ParBilinearForm(sfes);
        bf->AddDomainIntegrator(new mfem::MassIntegrator());
        bf->AddDomainIntegrator(new DiffusionIntegrator(dc));
        bf->Assemble();
        bf->Finalize();
        K=bf->ParallelAssemble();
        delete bf;

        A=K->EliminateRowsCols(ess_tdofv);
        K->EliminateZeroRows();

        //allocate the CG solver and the preconditioner
        prec=new mfem::HypreBoomerAMG(*K);
        pcg=new mfem::CGSolver(pmesh->GetComm());
        pcg->SetOperator(*K);
        pcg->SetPreconditioner(*prec);

        mfem::ParMixedBilinearForm* mf=new mfem::ParMixedBilinearForm(dfes,sfes);
        mf->AddDomainIntegrator(new mfem::MassIntegrator());
        mf->Assemble();
        mf->Finalize();
        S=mf->ParallelAssemble();
        delete mf;

    }

    //forward filter
    void FFilter(Coefficient* coeff, ParGridFunction& gf)
    {

        gf.SetSpace(GetFilterFES());
        tmpv.SetSize(GetFilterFES()->TrueVSize()); tmpv=0.0;
        rhsv.SetSize(GetFilterFES()->TrueVSize());

        ParLinearForm lf(GetFilterFES());
        lf.AddDomainIntegrator(new DomainLFIntegrator(*coeff));
        lf.Assemble();
        lf.ParallelAssemble(rhsv);

        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);

        K->EliminateBC(*A,ess_tdofv,bdrc,rhsv);
        pcg->Mult(rhsv,tmpv);
        gf.SetFromTrueDofs(tmpv);
    }

    void AFilter(Coefficient* coeff, ParGridFunction& gf)
    {
        gf.SetSpace(GetFilterFES());
        tmpv.SetSize(GetFilterFES()->TrueVSize()); tmpv=0.0;
        rhsv.SetSize(GetFilterFES()->TrueVSize());

        ParLinearForm lf(GetFilterFES());
        lf.AddDomainIntegrator(new DomainLFIntegrator(*coeff));
        lf.Assemble();
        lf.ParallelAssemble(rhsv);

        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);

        K->EliminateBC(*A,ess_tdofv,tmpv,rhsv);
        pcg->Mult(rhsv,tmpv);
        gf.SetFromTrueDofs(tmpv);
    }

private:
    mutable mfem::HypreParMatrix* S;
    mutable mfem::HypreParMatrix* K;
    mutable mfem::HypreParMatrix* A;
    mutable mfem::Solver* prec;
    mutable mfem::CGSolver* pcg;

    mfem::FiniteElementCollection* sfec;
    mfem::ParFiniteElementSpace* sfes;
    mfem::FiniteElementCollection* ifec;
    mfem::ParFiniteElementSpace* ifes;

    mfem::ParGridFunction sol;

    mutable mfem::Vector tmpv;
    mutable mfem::Vector bdrc;//boundary conditions
    mutable mfem::Vector rhsv;//RHS for the adjoint
    mutable mfem::Array<int> ess_tdofv; //boundary dofs

    double r;
    int order;

    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* dfes;

    std::map<int, mfem::ConstantCoefficient> bcr;

    double atol;
    double rtol;
    int max_iter;
    int prt_level;

};

class IsoComplCoef:public Coefficient
{
public:

    IsoComplCoef(GridFunction* rho_, GridFunction* sol_,bool SIMP_=false,bool PROJ_=false){

        eta=0.5;
        beta=8.0;
        p=1.0;

        SIMP=SIMP_;
        PROJ=PROJ_;

        rho=rho_;
        sol=sol_;

        dMu.reset(new DerivedCoef(this,&IsoComplCoef::EvalMu));
        dLambda.reset(new DerivedCoef(this,&IsoComplCoef::EvalLambda));
        dE.reset(new DerivedCoef(this,&IsoComplCoef::EvalE));
        dIsoCompl.reset(new DerivedCoef(this,&IsoComplCoef::EvalGrad));
    }

    virtual
    ~IsoComplCoef()
    {

    }

    void SetGridFunctions(GridFunction* rho_, GridFunction* sol_)
    {
        rho=rho_;
        sol=sol_;
    }

    void SetMaterial(real_t Emin_, real_t Emax_, real_t nu_)
    {
        cEmin.constant=Emin_;
        cEmax.constant=Emax_;
        cnu.constant=nu_;
        SetMaterial(&cEmin,&cEmax,&cnu);
    }

    void SetMaterial(Coefficient* Emin_, Coefficient* Emax_, Coefficient* nu_){
        Emin=Emin_;
        Emax=Emax_;
        nu=nu_;

        llmax.reset(new IsoElasticyLambdaCoeff(Emax,nu));
        llmin.reset(new IsoElasticyLambdaCoeff(Emin,nu));
        mmmax.reset(new IsoElasticySchearCoeff(Emax,nu));
        mmmin.reset(new IsoElasticySchearCoeff(Emin,nu));

    }

    void SetProj(double eta_, double beta_)
    {
        PROJ=true;
        eta=eta_;
        beta=beta_;
    }

    void SetSIMP(double p_)
    {
        SIMP=true;
        p=p_;
    }

    Coefficient* GetE(){return dE.get();}
    Coefficient* GetLambda(){return dLambda.get();}
    Coefficient* GetMu(){return dMu.get();}
    Coefficient* GetGradIsoComp(){return dIsoCompl.get();}


    virtual
    real_t Eval(ElementTransformation &T,
             const IntegrationPoint &ip) override
    {
        real_t Lmax = llmax->Eval(T, ip);
        real_t Mmax = mmmax->Eval(T, ip);

        real_t Lmin = llmin->Eval(T, ip);
        real_t Mmin = mmmin->Eval(T, ip);

        sol->GetVectorGradient(T, grad);
        real_t div_u = grad.Trace();
        real_t density_max = Lmax*div_u*div_u;
        real_t density_min = Lmin*div_u*div_u;

        int dim = T.GetSpaceDim();
        for (int i=0; i<dim; i++)
        {
            for (int j=0; j<dim; j++)
            {
                density_max += Mmax*grad(i,j)*(grad(i,j)+grad(j,i));
                density_min += Mmin*grad(i,j)*(grad(i,j)+grad(j,i));
            }
        }
        real_t val = rho->GetValue(T,ip);
        if(PROJ){
            val=PointwiseTrans::HProject(val,eta,beta);
        }

        if(SIMP){
            val=PointwiseTrans::SIMPInterpolation(val,p);
        }

        return val*density_max+density_min;
    }

private:

    real_t EvalGrad(ElementTransformation &T,
             const IntegrationPoint &ip)
    {
        real_t Lmax = llmax->Eval(T, ip);
        real_t Mmax = mmmax->Eval(T, ip);


        sol->GetVectorGradient(T, grad);
        real_t div_u = grad.Trace();
        real_t density_max = Lmax*div_u*div_u;

        int dim = T.GetSpaceDim();
        for (int i=0; i<dim; i++)
        {
            for (int j=0; j<dim; j++)
            {
                density_max += Mmax*grad(i,j)*(grad(i,j)+grad(j,i));
            }
        }
        real_t val = rho->GetValue(T,ip);
        real_t hvl = val;
        real_t gvl=1.0;

        if(PROJ){
            hvl=PointwiseTrans::HProject(val,eta,beta);
        }

        if(SIMP){
            gvl=gvl*PointwiseTrans::GradSIMPInterpolation(hvl,p);
        }

        if(PROJ){
            gvl=gvl*PointwiseTrans::HGrad(val,eta,beta);
        }

        return gvl*density_max;
    }

    real_t EvalLambda(ElementTransformation &T,
                      const IntegrationPoint &ip)
    {
        real_t Lmax = llmax->Eval(T, ip);
        real_t Lmin = llmin->Eval(T, ip);

        real_t val = rho->GetValue(T,ip);
        if(PROJ){
            val=PointwiseTrans::HProject(val,eta,beta);
        }

        if(SIMP){
            val=PointwiseTrans::SIMPInterpolation(val,p);
        }

        return Lmax*val+Lmin;
    }

    real_t EvalMu(ElementTransformation &T,
                  const IntegrationPoint &ip)
    {
        real_t Mmax = mmmax->Eval(T, ip);
        real_t Mmin = mmmin->Eval(T, ip);

        real_t val = rho->GetValue(T,ip);
        if(PROJ){
            val=PointwiseTrans::HProject(val,eta,beta);
        }

        if(SIMP){
            val=PointwiseTrans::SIMPInterpolation(val,p);
        }

        return Mmax*val+Mmin;
    }

    real_t EvalE(ElementTransformation &T,
             const IntegrationPoint &ip)
    {
        real_t vEmax = Emax->Eval(T,ip);
        real_t vEmin = Emin->Eval(T,ip);
        real_t val = rho->GetValue(T,ip);
        if(PROJ){
            val=PointwiseTrans::HProject(val,eta,beta);
        }

        if(SIMP){
            val=PointwiseTrans::SIMPInterpolation(val,p);
        }

        return vEmax*val+vEmin;
    }

    real_t eta;
    real_t beta;
    real_t p;

    bool SIMP=0;
    bool PROJ=0;

    Coefficient* Emax;
    Coefficient* Emin;
    Coefficient* nu;

    ConstantCoefficient cEmax;
    ConstantCoefficient cEmin;
    ConstantCoefficient cnu;

    std::unique_ptr<IsoElasticyLambdaCoeff> llmax;
    std::unique_ptr<IsoElasticySchearCoeff> mmmax;

    std::unique_ptr<IsoElasticyLambdaCoeff> llmin;
    std::unique_ptr<IsoElasticySchearCoeff> mmmin;

    GridFunction* sol;
    GridFunction* rho;

    DenseMatrix grad;


    real_t (IsoComplCoef::*ptr2grad)(ElementTransformation &T,
                                     const IntegrationPoint &ip) = &IsoComplCoef::EvalGrad;

    real_t (IsoComplCoef::*ptr2Mu)(ElementTransformation &T,
                                     const IntegrationPoint &ip) = &IsoComplCoef::EvalMu;

    real_t (IsoComplCoef::*ptr2Lambda)(ElementTransformation &T,
                                   const IntegrationPoint &ip) = &IsoComplCoef::EvalLambda;

    real_t (IsoComplCoef::*ptr2E)(ElementTransformation &T,
                                       const IntegrationPoint &ip) = &IsoComplCoef::EvalE;


    class DerivedCoef:public Coefficient
    {
    public:
        DerivedCoef(IsoComplCoef* obj_, real_t (IsoComplCoef::*methodPtr)(ElementTransformation &T,
                                                                       const IntegrationPoint &ip))
        {
            obj=obj_;
            ptr2eval=methodPtr;
        }

        virtual
            real_t Eval(ElementTransformation &T,
                 const IntegrationPoint &ip) override
        {
            return (obj->*ptr2eval)(T,ip);
        }

    private:
        IsoComplCoef* obj;
        real_t (IsoComplCoef::*ptr2eval)(ElementTransformation &T,
                                         const IntegrationPoint &ip);
    };

    std::unique_ptr<DerivedCoef> dMu;
    std::unique_ptr<DerivedCoef> dE;
    std::unique_ptr<DerivedCoef> dLambda;
    std::unique_ptr<DerivedCoef> dIsoCompl;

};

#endif
