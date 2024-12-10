#ifndef REMAP_OPT_H
#define REMAP_OPT_H

#include "mfem.hpp"

class SphCoefficient:public mfem::Coefficient
{
public:
    SphCoefficient(mfem::real_t r_=0.30):r(r_)
    {

    }

    ~SphCoefficient()
    {


    }

    virtual
    mfem::real_t Eval(mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip)
    {
        mfem::Vector tmpv;
        tmpv.SetSize(T.GetSpaceDim());
        T.Transform(ip,tmpv);

        for(int i=0;i<tmpv.Size();i++){
            tmpv[i]-=0.40;
        }

        mfem::real_t rez=tmpv.Norml2();


        if(rez<r){return 1.0;}
        else{return 0.0;}
    }


private:
    mfem::real_t r;
};


class L2Objective:public mfem::Operator
{
public:
    L2Objective(mfem::ParFiniteElementSpace& pfes_,
                mfem::ParGridFunction& tgf_):cgf(tgf_),tgf(&tgf_)
    {
        bf=new mfem::ParBilinearForm(&pfes_);
        bf->AddDomainIntegrator(new mfem::MassIntegrator());
        bf->Assemble();

        tgv.SetSize(pfes_.GetTrueVSize());
        bf->Mult(tgf->GetTrueVector(),tgv);

    }

    ~L2Objective()
    {
        delete bf;
    }

    mfem::real_t Eval(mfem::Vector& x)
    {
        cgf.SetFromTrueDofs(x);
        gfc.SetGridFunction(&cgf);
        mfem::real_t vv=tgf->ComputeL2Error(gfc);
        return 0.5*vv*vv;
    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        bf->Mult(x,y);
        y-=tgv;
    }

    void Test()
    {
        mfem::ConstantCoefficient cc(0.5);
        mfem::ParGridFunction lgf(*tgf);
        lgf.ProjectCoefficient(cc);

        mfem::Vector x=lgf.GetTrueVector();
        mfem::Vector p; p.SetSize(x.Size()); p.Randomize();
        mfem::Vector g; g.SetSize(x.Size());
         mfem::Vector tmpv; tmpv.SetSize(x.Size());

        double lo=this->Eval(x);
        this->Mult(x,g);

        mfem::real_t nd=mfem::InnerProduct(cgf.ParFESpace()->GetComm(),p,p);
        mfem::real_t td=mfem::InnerProduct(cgf.ParFESpace()->GetComm(),p,g);

        td=td/nd;

        double lsc=1.0;
        double lqoi;

        for(int l=0;l<10;l++){
           lsc/=10.0;
           p/=10.0;

           add(p,x,tmpv);
           lqoi=this->Eval(tmpv);
           mfem::real_t ld=(lqoi-lo)/lsc;
           if(cgf.ParFESpace()->GetMyRank()==0){
             std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                        << " gradient=" << td
                        << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
        }
    }


private:
    mfem::ConstantCoefficient one;
    mfem::ParGridFunction cgf;
    mfem::ParGridFunction* tgf;
    mfem::ParBilinearForm* bf;
    mfem::GridFunctionCoefficient gfc;

    mfem::Vector tgv;

};

class VolConstr:public mfem::Operator
{
public:
    VolConstr(mfem::ParFiniteElementSpace& pfes_, //fes of the density field
              double vol_): //target volume
                pfes(&pfes_),vol(vol_)
    {


        l=new mfem::ParLinearForm(pfes);
        l->AddDomainIntegrator(new mfem::DomainLFIntegrator(one,pfes->GetMaxElementOrder()));
        l->Assemble();

        gf=new mfem::ParGridFunction(pfes);

    }

    ~VolConstr()
    {
        delete l;
        delete gf;
    }


    //input x:true dofs
    double Eval(mfem::Vector& x)
    {
        gf->SetFromTrueDofs(x);
        return (*l)(*gf)-vol;
    }

    virtual
    void Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        l->ParallelAssemble(y);
    }

    void Test()
    {
        mfem::ConstantCoefficient cc(0.5);
        mfem::ParGridFunction lgf(pfes);
        lgf.ProjectCoefficient(cc);

        mfem::Vector x=lgf.GetTrueVector();
        mfem::Vector p; p.SetSize(x.Size()); p.Randomize();
        mfem::Vector g; g.SetSize(x.Size());
         mfem::Vector tmpv; tmpv.SetSize(x.Size());

        double lo=this->Eval(x);
        this->Mult(x,g);

        mfem::real_t nd=mfem::InnerProduct(pfes->GetComm(),p,p);
        mfem::real_t td=mfem::InnerProduct(pfes->GetComm(),p,g);

        td=td/nd;

        double lsc=1.0;
        double lqoi;

        for(int l=0;l<10;l++){
           lsc/=10.0;
           p/=10.0;
           add(p,x,tmpv);
           lqoi=this->Eval(tmpv);
           mfem::real_t ld=(lqoi-lo)/lsc;
           if(pfes->GetMyRank()==0){
             std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                        << " gradient=" << td
                        << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
        }
    }




private:
    mfem::ConstantCoefficient one;
    mfem::ParFiniteElementSpace* pfes;
    double vol;

    mfem::ParLinearForm* l;
    mfem::ParGridFunction* gf;

};

class MDSolver
{
public:
    MDSolver(mfem::ParFiniteElementSpace& pfes,
         mfem::real_t vol_,
         mfem::ParGridFunction& trg,
         mfem::Vector& u_min_,
         mfem::Vector& u_max_):cfg(&pfes),tfg(&pfes)
    {
       tfg=u_max_; tfg.GetTrueDofs(u_max);
       tfg=u_min_; tfg.GetTrueDofs(u_min);

       vol=vol_;
       obj=new L2Objective(pfes,trg);

       con=new VolConstr(pfes,vol);

       //initialize the current solution
       for(int i=0;i<cfg.Size();i++){
        cfg[i]=0.5*(u_min_[i]+u_max_[i]);
       }
    }

    ~MDSolver()
    {
        delete obj;
        delete con;
    }

    mfem::ParGridFunction& GetTFG(){return tfg;}
    mfem::ParGridFunction& GetCFG(){return cfg;}

    void SetFinal(mfem::ParGridFunction& gf)
    {
        gf.SetFromTrueDofs(cfg.GetTrueVector());
    }

    void Optimize(double alpha, double rho,int max_iter=100)
    {

        mfem::Vector x; x.SetSize(u_max.Size()); cfg.GetTrueDofs(x);
        mfem::Vector p; p.SetSize(x.Size()); p=0.0;
        //initialize the current solution
        for(int i=0;i<x.Size();i++){
            //p[i]=std::log((x[i]-u_min[i])/(u_max[i]-x[i]));
            x[i]=(u_min[i]+u_max[i]*std::exp(p[i]))/(1.0+std::exp(p[i]));
        }

        mfem::real_t epsp=obj->Eval(x);
        mfem::real_t epsc=con->Eval(x);
        mfem::real_t epso; epso=epsp;

        if(cfg.ParFESpace()->GetMyRank()==0){
            std::cout<<" epsp="<<epsp<<" epsc="<<epsc<<std::endl;
        }

        mfem::real_t lambda=0.0;

        mfem::Vector go,gp;
        go.SetSize(x.Size());
        gp.SetSize(x.Size());

        bool flag=true;

        int it=0;
        while(flag){
            //evaluate the gradients
            obj->Mult(x,go);


            con->Mult(x,gp);
            go.Add(lambda,gp);
            go.Add(rho*epsc,gp);

            //update p
            p.Add(-alpha,go);
            //update x
            for(int i=0;i<x.Size();i++){
                x[i]=(u_min[i]+u_max[i]*std::exp(p[i]))/(1.0+std::exp(p[i]));
            }
            //evaluate the objective and the contraint
            epsp=obj->Eval(x);
            epsc=con->Eval(x);
            //update lambda
            lambda=lambda+rho*epsc;

            if(cfg.ParFESpace()->GetMyRank()==0){
                std::cout<<" epsp="<<epsp<<" epsc="<<epsc<<" lambda="<<lambda<<std::endl;
            }

            if(fabs(epsc)<1e-10){
                if(fabs(epsp-epso)<1e-10){
                    flag=false;
                }
            }
            epso=epsp;
            it++;
            if(it>max_iter){ flag=false;}
        }

        cfg.SetFromTrueDofs(x);
        tfg.SetFromTrueDofs(p);


    }

private:
    mfem::ParGridFunction cfg;
    mfem::ParGridFunction tfg;
    mfem::Vector u_max;
    mfem::Vector u_min;
    mfem::real_t vol;
    L2Objective* obj;
    VolConstr* con;
};


namespace mfem{

namespace Opt{

class Functional:public Operator
{
public:

    Functional(int s):Operator(s)
    {

    }

    virtual
    ~Functional();

    /// Evaluate the functional for a given set of values.
    /// For FEM fields the expectation is that x is the set
    /// of true DOFs.
    virtual
    real_t Eval(const Vector& x)=0;

    /// Evaluate the gradient of the functional for a
    /// given set of values stored in vector x. For FEM
    /// fields the expectation is that x is the set of
    /// true DOFs.
    virtual
    void CalcGrad(const Vector&x, Vector& grad) const =0;


    /// The mult method inherited from Operator calls
    /// CalcGrad and allows the functional to be used with
    /// MFEM linear and non-linear solvers.
    virtual
    void Mult(const Vector& x, Vector& y) const override
    {
        CalcGrad(x,y);
    }

    /// Evaluate the product of the Hessian of the functional,
    /// for parameter set stored in x, with vector y (rpimal space) and
    /// the result is stored in vector r (dual space).
    virtual
    void HessianMult(const Vector &x, const Vector& y, Vector& r)
    {
        mfem_error("Opt::Functional::HessianMult() is not overridden!");
    }

    /// Evaluate the product of the transpose Hessian of the functional,
    /// for parameter set stored in x, with vector y (dual space) and
    /// the result is stored in vector r (primal space).
    virtual
    void HessianMultTranspose(const Vector &x, const Vector& y, Vector& r)
    {
        mfem_error("Opt::Functional::HessianMult() is not overridden!");
    }

private:

};


class OptimizationProblem:public Functional
{
public:
    enum ConstrType{EQUALITY, INEQUALITY};

    OptimizationProblem(int s=0, int num_eq=0, int num_ineq=0):Functional(s)
    {
        EqConstr.resize(num_eq);
        InConstr.resize(num_ineq);
    }

    virtual
    int GetNumConstr(ConstrType ctyp){
        if(ctyp==EQUALITY){
            return EqConstr.size();
        }else
        if(ctyp==INEQUALITY)
        {
            return InConstr.size();
        }else{ return 0;}

    };


    void SetConstr(int i, std::shared_ptr<Functional> fun,ConstrType ctyp=EQUALITY)
    {
        if(ctyp==EQUALITY)
        {
            EqConstr[i]=fun;
        }else
        if(ctyp==INEQUALITY)
        {
            InConstr[i]=fun;
        }
    }

    virtual
    std::shared_ptr<Functional> GetConstr(int i, ConstrType ctyp=EQUALITY)
    {
        if(ctyp==EQUALITY)
        {
            return EqConstr[i];
        }else
        if(ctyp==INEQUALITY)
        {
            return InConstr[i];
        }
    }

    virtual
    void CalcConstrGrad(int i, ConstrType ctyp, const Vector&x, Vector& grad)
    {
        if(ctyp==EQUALITY){
            EqConstr[i]->CalcGrad(x,grad);
        }else
        if(ctyp==INEQUALITY)
        {
            InConstr[i]->CalcGrad(x,grad);
        }
    }


    virtual
    void ConstrHessianMult(int i, ConstrType ctyp, const Vector &x, const Vector& y, Vector& r)
    {
        if(ctyp==EQUALITY){
            EqConstr[i]->HessianMult(x,y,r);
        }else
        if(ctyp==INEQUALITY)
        {
            InConstr[i]->HessianMult(x,y,r);
        }
    }

    virtual
    void ConstrHessianMultTranspose(int i, ConstrType ctyp,
                                    const Vector &x, const Vector& y, Vector& r)
    {
        if(ctyp==EQUALITY){
            EqConstr[i]->HessianMultTranspose(x,y,r);
        }else
        if(ctyp==INEQUALITY)
        {
            InConstr[i]->HessianMultTranspose(x,y,r);
        }
    }

    virtual
    ~OptimizationProblem();

private:

    std::vector< std::shared_ptr<Functional> > EqConstr;
    std::vector< std::shared_ptr<Functional> > InConstr;

};

}


}


#endif
