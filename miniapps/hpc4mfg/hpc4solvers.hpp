#ifndef HPC4SOLVERS_HPP
#define HPC4SOLVERS_HPP

#include "mfem.hpp"

namespace mfem{


class BasicNLDiffusionCoefficient{
public:


    virtual
    ~BasicNLDiffusionCoefficient(){}

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)=0;

    virtual
    void Grad(ElementTransformation &T,
              const IntegrationPoint &ip,
              const Vector& u, Vector& r)=0;

    virtual
    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)=0;

private:
};


class ExampleNLDiffusionCoefficient:public BasicNLDiffusionCoefficient
{
public:

    ExampleNLDiffusionCoefficient(double a_=1.0, double b_=1.0, double c_=0.0, double d_=0.1){
        ownership=true;
        a=new ConstantCoefficient(a_);
        b=new ConstantCoefficient(b_);
        c=new ConstantCoefficient(c_);
        d=new ConstantCoefficient(d_);
    }

    ExampleNLDiffusionCoefficient(Coefficient& a_, Coefficient& b_,
                                  Coefficient& c_, Coefficient& d_)
    {
        ownership=false;
        a=&a_;
        b=&b_;
        c=&c_;
        d=&d_;
    }

    virtual
    ~ExampleNLDiffusionCoefficient()
    {
        if(ownership)
        {
            delete a;
            delete b;
            delete c;
            delete d;
        }
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);

        return 0.5*aa*(du*du)+0.5*bb*(u[dim]-cc)*(u[dim]-cc)+0.5*dd*exp(du*du);
    }

    virtual
    void Grad(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u, Vector& r)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);
        double nd=du*du;

        r.SetSize(dim+1); r=0.0;

        for(int i=0; i<dim; i++){
            r[i]=aa*du[i]+dd*exp(nd)*du[i];
        }
        r[dim]=bb*(u[dim]-cc);
    }

    virtual
    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);
        double nd=du*du;

        h.SetSize(dim+1,dim+1); h=0.0;

        for(int i=0;i<dim;i++){
            h(i,i)=aa+dd*exp(nd);
            for(int j=0;j<dim;j++){
                h(i,j)=h(i,j)+2.0*dd*exp(nd)*du[i]*du[j];
            }
        }

    }


private:
    bool ownership;
    Coefficient* a;
    Coefficient* b;
    Coefficient* c;
    Coefficient* d;

    double aa;
    double bb;
    double cc;
    double dd;

};


class NLDiffusionIntegrator:public NonlinearFormIntegrator
{
public:
    NLDiffusionIntegrator()
    {
        mat=nullptr;
    }

    NLDiffusionIntegrator(BasicNLDiffusionCoefficient* mat_)
    {
        mat=mat_;
    }


    void SetMaterial(BasicNLDiffusionCoefficient* mat_)
    {
        mat=mat_;
    }

    virtual
    ~NLDiffusionIntegrator(){}

    virtual
    double GetElementEnergy(const FiniteElement &el,
                            ElementTransformation &trans,
                            const Vector &elfun);


    virtual
    void AssembleElementVector(const FiniteElement &el,
                               ElementTransformation &trans,
                               const Vector &elfun,
                               Vector &elvect);


    virtual
    void AssembleElementGrad(const FiniteElement &el,
                             ElementTransformation &trans,
                             const Vector &elfun,
                             DenseMatrix &elmat);
private:

    BasicNLDiffusionCoefficient* mat;
};


class NLDiffusion{
public:
    NLDiffusion(mfem::ParMesh* mesh_, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();
        fec=new H1_FECollection(order_,dim);
        fes=new ParFiniteElementSpace(pmesh,fec);

        sol.SetSize(fes->GetTrueVSize()); sol=0.0;
        rhs.SetSize(fes->GetTrueVSize()); rhs=0.0;
        adj.SetSize(fes->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(fes);
        adjgf.SetSpace(fes);

        nf=nullptr;
        SetNewtonSolver();
        SetLinearSolver();


        prec=nullptr;
        ls=nullptr;
        ns=nullptr;
    }

    ~NLDiffusion(){
        delete ns;
        delete ls;
        delete prec;
        delete nf;
        delete fes;
        delete fec;

        for(size_t i=0;i<materials.size();i++){
            delete materials[i];
        }
    }

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, double val)
    {
        bc[id]=mfem::ConstantCoefficient(val);
        AddDirichletBC(id,bc[id]);
    }

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, mfem::Coefficient& val)
    {
        bcc[id]=&val;
    }

    /// Adds Neumann BC
    void AddNeumannBC(int id, double val)
    {
        nc[id]=mfem::ConstantCoefficient(val);
        AddNeumannBC(id,nc[id]);
    }

    /// Adds Neumann BC
    void AddNeumannBC(int id, mfem::Coefficient& val)
    {
        ncc[id]=&val;
    }

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the adjoint solution
    mfem::ParGridFunction& GetAdjoint(){return adjgf;}

    /// Add material to the solver. The pointer is owned by the solver.
    void AddMaterial(BasicNLDiffusionCoefficient* nmat)
    {
        materials.push_back(nmat);
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(fes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(fes); agf.SetFromTrueDofs(adj);}

private:
    mfem::ParMesh* pmesh;

    std::vector<BasicNLDiffusionCoefficient*> materials;

    //solution true vector
    mfem::Vector sol;
    //adjoint true vector
    mfem::Vector adj;
    //RHS
    mfem::Vector rhs;


    mfem::ParGridFunction solgf;
    mfem::ParGridFunction adjgf;


    mfem::FiniteElementCollection *fec;
    mfem::FiniteElementSpace	  *fes;
    mfem::ParNonlinearForm *nf;

    //Newton solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;


    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    mfem::HypreBoomerAMG *prec; //preconditioner
    mfem::CGSolver *ls;  //linear solver
    mfem::NewtonSolver *ns;

    // holds DBC in coefficient form
    std::map<int, mfem::Coefficient*> bcc;

    // holds internal DBC
    std::map<int, mfem::ConstantCoefficient> bc;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    // holds internal NBC
    std::map<int, mfem::ConstantCoefficient> nc;

    mfem::Array<int> ess_tdofv;

};





}

#endif
