#ifndef MTOP_FILTERS_HPP
#define MTOP_FILTERS_HPP

#include "mfem.hpp"

namespace mfem {

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


}


class FilterSolver
{
public:
    FilterSolver(double r_, mfem::ParMesh* pmesh_, int order_=2)
    {
        r=r_;
        order=order_;
        pmesh=pmesh_;
        int dim=pmesh->Dimension();
        sfec=new mfem::H1_FECollection(order, dim);
        sfes=new mfem::ParFiniteElementSpace(pmesh,sfec,1);

        ifec=new mfem::H1Pos_FECollection(order,dim);
        ifes=new mfem::ParFiniteElementSpace(pmesh,ifec,1);

        dfes=ifes;
        SetSolver();


        K=nullptr;
        S=nullptr;
        A=nullptr;
        pcg=nullptr;
        prec=nullptr;
    }

    FilterSolver(double r_, mfem::ParMesh* pmesh_, mfem::ParFiniteElementSpace* dfes_,int order_=2):dfes(dfes_)
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

    }

    mfem::ParFiniteElementSpace* GetFilterFES(){return sfes;}


    virtual
    ~FilterSolver()
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
        AllocSolvers();
    }

    virtual
    void Mult(const Vector &x, Vector &y)
    {
        if(pcg==nullptr){
            AllocSolvers();
        }

        y=bdrc;
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
    void MultTranspose(const Vector &x, Vector &y)
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

    void AllocSolvers()
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


private:
    mfem::HypreParMatrix* S;
    mfem::HypreParMatrix* K;
    mfem::HypreParMatrix* A;
    mfem::Solver* prec;
    mfem::CGSolver* pcg;

    mfem::FiniteElementCollection* sfec;
    mfem::ParFiniteElementSpace* sfes;
    mfem::FiniteElementCollection* ifec;
    mfem::ParFiniteElementSpace* ifes;

    mfem::Vector tmpv;
    mfem::Vector bdrc;//boundary conditions
    mfem::Vector rhsv;//RHS for the adjoint
    mfem::Array<int> ess_tdofv; //boundary dofs

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


class PVolumeQoIIntegrator:public NonlinearFormIntegrator
{
public:
    PVolumeQoIIntegrator(double eta_, double beta_, int iorder_=4)
    {
        eta=eta_;
        beta=beta_;
        iorder=iorder_;
    }

    virtual
    ~PVolumeQoIIntegrator()
    {

    }

    virtual
    double GetElementEnergy(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun) override
    {
        double energy=0.0;
        const int ndof = el.GetDof();
        const int ndim = el.GetDim();

        const mfem::IntegrationRule *ir = NULL;
        int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        double w;
        double tval;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            tval=shapef*elfun;
            //trim the density for high-order fields
            if(tval>1.0){tval=1.0;}
            else if(tval<0.0){tval=0.0;}
            w= mfem::PointwiseTrans::HProject(tval,eta,beta);
            w= ip.weight * trans.Weight() * w;
            energy = energy + w;
        }
        return energy;
    }

    virtual
    void AssembleElementVector(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const mfem::Vector & elfun,
                                       mfem::Vector & elvect) override
    {

        const int ndof = el.GetDof();

        const mfem::IntegrationRule *ir = NULL;
        int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        elvect.SetSize(ndof);
        elvect=0.0;

        mfem::Vector shapef(ndof);
        double w;
        double tval;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            tval=shapef*elfun;
            //trim the density for high-order fields
            if(tval>1.0){tval=1.0;}
            else if(tval<0.0){tval=0.0;}

            w= mfem::PointwiseTrans::HGrad(tval,eta,beta);
            w= ip.weight * trans.Weight() * w;
            elvect.Add(w,shapef);
        }

    }

    virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                      mfem::ElementTransformation & trans,
                                      const mfem::Vector & elfun,
                                      mfem::DenseMatrix & elmat) override
     {
         const int ndof = el.GetDof();

         const mfem::IntegrationRule *ir = NULL;
         int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);

         elmat.SetSize(ndof);
         elmat=0.0;

         mfem::Vector shapef(ndof);
         double w;
         double tval;
         for (int i = 0; i < ir -> GetNPoints(); i++)
         {
             const mfem::IntegrationPoint &ip = ir->IntPoint(i);
             trans.SetIntPoint(&ip);
             el.CalcShape(ip,shapef);
             tval=shapef*elfun;
             //trim the density for high-order fields
             if(tval>1.0){tval=1.0;}
             else if(tval<0.0){tval=0.0;}
             w=mfem::PointwiseTrans::HHess(tval,eta,beta);
             w= ip.weight * trans.Weight() * w;
             AddMult_a_VVt(w, shapef, elmat);
         }

    }


private:
    double eta;
    double beta;
    int iorder;

};


class VolumeQoIIntegrator:public NonlinearFormIntegrator
{
public:
    VolumeQoIIntegrator(int iorder_=4)
    {
        iorder=iorder_;
    }

    virtual
    ~VolumeQoIIntegrator()
    {

    }

    virtual
    double GetElementEnergy(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun) override
    {
        double energy=0.0;
        const int ndof = el.GetDof();
        const int ndim = el.GetDim();

        const mfem::IntegrationRule *ir = NULL;
        int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        double w;
        double tval;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            tval=shapef*elfun;
            //trim the density for high-order fields
            if(tval>1.0){tval=1.0;}
            else if(tval<0.0){tval=0.0;}
            w= tval;
            w= ip.weight * trans.Weight() * w;
            energy = energy + w;
        }
        return energy;
    }

    virtual
    void AssembleElementVector(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const mfem::Vector & elfun,
                                       mfem::Vector & elvect) override
    {

        const int ndof = el.GetDof();

        const mfem::IntegrationRule *ir = NULL;
        int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        elvect.SetSize(ndof);
        elvect=0.0;

        mfem::Vector shapef(ndof);
        double w;
        double tval;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            tval=shapef*elfun;
            //trim the density for high-order fields
            if(tval>1.0){tval=1.0;}
            else if(tval<0.0){tval=0.0;}

            w= 1.0;
            w= ip.weight * trans.Weight() * w;
            elvect.Add(w,shapef);
        }

    }

    virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                      mfem::ElementTransformation & trans,
                                      const mfem::Vector & elfun,
                                      mfem::DenseMatrix & elmat) override
     {
         const int ndof = el.GetDof();

         const mfem::IntegrationRule *ir = NULL;
         int order = iorder * trans.OrderGrad(&el) - 1; // correct order?
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);

         elmat.SetSize(ndof);
         elmat=0.0;

         mfem::Vector shapef(ndof);
         double w;
         double tval;
         for (int i = 0; i < ir -> GetNPoints(); i++)
         {
             const mfem::IntegrationPoint &ip = ir->IntPoint(i);
             trans.SetIntPoint(&ip);
             el.CalcShape(ip,shapef);
             tval=shapef*elfun;
             //trim the density for high-order fields
             if(tval>1.0){tval=1.0;}
             else if(tval<0.0){tval=0.0;}
             w=0.0;
             w= ip.weight * trans.Weight() * w;
             AddMult_a_VVt(w, shapef, elmat);
         }

    }


private:
    double eta;
    double beta;
    int iorder;

};


class VolumeQoI{
public:
    VolumeQoI(mfem::ParFiniteElementSpace* fes_)
    {
        fes=fes_;
        nf=nullptr;
    }

    ~VolumeQoI()
    {
        delete nf;
    }

    void SetProjection(double eta_,double beta_)
    {
    }

    void Update()
    {
        delete nf;
        nf=nullptr;
        fes->Update();
    }

    /// Input: true design vector
    double Eval(mfem::Vector& design_)
    {
        Alloc();
        return nf->GetEnergy(design_);
    }

    void Grad(mfem::Vector& design_,mfem::Vector& grad)
    {
        Alloc();
        nf->Mult(design_,grad);
    }

private:
    mfem::ParNonlinearForm* nf;
    mfem::ParFiniteElementSpace* fes;

    void Alloc()
    {
        if(nf==nullptr)
        {
            nf=new mfem::ParNonlinearForm(fes);
            nf->AddDomainIntegrator(new mfem::VolumeQoIIntegrator());
        }
    }
};

//computes the volume of a projected field
class PVolumeQoI
{
public:
    PVolumeQoI(mfem::ParFiniteElementSpace* fes_)
    {
        fes=fes_;
        nf=nullptr;
        SetProjection(0.5,8.0);
    }

    ~PVolumeQoI()
    {
        delete nf;
    }

    void SetProjection(double eta_,double beta_)
    {
        eta=eta_;
        beta=beta_;
        delete nf;
        nf=nullptr;
    }

    void Update()
    {
        delete nf;
        nf=nullptr;
        fes->Update();
    }

    /// Input: true design vector
    double Eval(mfem::Vector& design_)
    {
        Alloc();
        return nf->GetEnergy(design_);
    }

    void Grad(mfem::Vector& design_,mfem::Vector& grad)
    {
        Alloc();
        nf->Mult(design_,grad);
    }

private:
    double eta;
    double beta;
    mfem::ParNonlinearForm* nf;
    mfem::ParFiniteElementSpace* fes;

    void Alloc()
    {
        if(nf==nullptr)
        {
            nf=new mfem::ParNonlinearForm(fes);
            nf->AddDomainIntegrator(new mfem::PVolumeQoIIntegrator(eta,beta));
        }
    }
};

}

#endif
