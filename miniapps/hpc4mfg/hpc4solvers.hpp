#ifndef HPC4SOLVERS_HPP
#define HPC4SOLVERS_HPP

#include "mfem.hpp"
#include "ascii.hpp"

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

    ExampleNLDiffusionCoefficient(double a_=1.0, double b_=1.0, double c_=1.0, double d_=0.1){
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

class SurrogateNLDiffusionCoefficient:public BasicNLDiffusionCoefficient
{
public:

    SurrogateNLDiffusionCoefficient( 
        std::string & GradName,
        std::string & HessianName )

    {
        this->readSurrogateModel();
    }

    ~SurrogateNLDiffusionCoefficient(){}

    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)
    {
        return 0.0;
    }

    void Grad(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u, Vector& r)
    {
        int dim=u.Size()-1;
        mfem::Vector du(u.GetData(),dim);
        mfem::Vector HiddenVec( Weight1Rows );

        mWeightMatrix1.MultTranspose( du, HiddenVec );
        HiddenVec -= mBiasVector1;

        // activation function
        HiddenVec = HiddenVec;       //FIXME add activation function

        mfem::Vector OutputVec( Weight2Rows );
        mWeightMatrix2.MultTranspose( HiddenVec, OutputVec );
        OutputVec -= mBiasVector1;
    }

    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        int dim=u.Size()-1;
        mfem::Vector du(u.GetData(),dim);
        mfem::Vector HiddenVec( Weight1Rows );


        HessWeightMatrix1.MultTranspose( du, HiddenVec );
        HiddenVec -= HessBiasVector1;

        // activation function
        HiddenVec = HiddenVec;       //FIXME add activation function

        mfem::Vector OutputVec( Weight2Rows );
        HessWeightMatrix2.MultTranspose( HiddenVec, OutputVec );
        OutputVec -= HessBiasVector1;

        mfem::DenseMatrix LMat(dim,dim);  LMat = 0.0;
        mfem::DenseMatrix SkewMat(dim,dim);  SkewMat = 0.0;
        mfem::DenseMatrix LLT(dim,dim);  LLT = 0.0;

        // tril
        LMat(0,0) = OutputVec( 0 );
        LMat(1,0) = OutputVec( 1 );
        LMat(1,1) = OutputVec( 2 );
        LMat(2,0) = OutputVec( 3 );
        LMat(2,1) = OutputVec( 4 );
        LMat(2,2) = OutputVec( 5 );

        MultAAt( LMat, LLT );

        // triu
        SkewMat(0,1) = OutputVec( 6 );
        SkewMat(0,2) = OutputVec( 7 );
        SkewMat(1,2) = OutputVec( 8 );

        mfem::DenseMatrix SkewMatTrans;
        SkewMatTrans.Transpose(SkewMat);

        h = 0.0;
        h +=LLT;
        h +=SkewMat;
        h -=SkewMatTrans;
    }

    void GradientVelWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, Vector& h)
    {

    }


private:

    mfem::DenseMatrix mWeightMatrix1;
    mfem::DenseMatrix mWeightMatrix2;
    mfem::Vector mBiasVector1;
    mfem::Vector mBiasVector2;

    int Weight1Rows;
    int Weight1Cols;
    int Weight2Rows;
    int Weight2Cols;

    mfem::DenseMatrix HessWeightMatrix1;
    mfem::DenseMatrix HessWeightMatrix2;
    mfem::Vector HessBiasVector1;
    mfem::Vector HessBiasVector2;

    int HessWeight1Rows;
    int HessWeight1Cols;
    int HessWeight2Rows;
    int HessWeight2Cols;

    void readSurrogateModel()
    {
        std::string tStringWeight1 = "./weights1.txt";
        std::string tStringWeight2 = "./weights2.txt";
        std::string tStringBias1   = "./bias1.txt";
        std::string tStringBias2 = "./bias2.txt";

        mfem::Ascii tAsciiReader1( tStringWeight1, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader2( tStringWeight2, FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader3( tStringBias1,   FileMode::OPEN_RDONLY );
        mfem::Ascii tAsciiReader4( tStringBias2,   FileMode::OPEN_RDONLY );

        int tNumLines1 = tAsciiReader1.length();        Weight1Rows = tNumLines1;
        int tNumLines2 = tAsciiReader2.length();        Weight2Rows = tNumLines2;
        int tNumLines3 = tAsciiReader3.length();
        int tNumLines4 = tAsciiReader4.length();

        int Weight1Cols = split_string(tAsciiReader1.line( 0 ), " ").size();
        int Weight2Cols = split_string(tAsciiReader2.line( 0 ), " ").size();

        mWeightMatrix1.SetSize(tNumLines1, Weight1Cols);
        mWeightMatrix2.SetSize(tNumLines2, Weight2Cols);
        mBiasVector1  .SetSize(tNumLines3);
        mBiasVector2  .SetSize(tNumLines4);

        for( int Ik = 0; Ik < tNumLines1; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader1.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                mWeightMatrix1( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines2; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader2.line( Ik );

            std::vector<std::string> ListOfStrings = split_string( tFileLine, " " );

            for( int Ii = 0; Ii < ListOfStrings.size(); Ii++ )
            {
                mWeightMatrix2( Ik, Ii ) = std::stod( ListOfStrings[Ii] );
            }
        }

        for( int Ik = 0; Ik < tNumLines3; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader4.line( Ik );
                
            mBiasVector1( Ik ) = std::stod( tFileLine );

        }

        for( int Ik = 0; Ik < tNumLines4; Ik++ )
        {
            const std::string & tFileLine = tAsciiReader4.line( Ik );

            mBiasVector2( Ik ) = std::stod( tFileLine );
        }
    }
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
    mfem::ParFiniteElementSpace	  *fes;
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
