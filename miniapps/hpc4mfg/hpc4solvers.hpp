#ifndef HPC4SOLVERS_HPP
#define HPC4SOLVERS_HPP

// #include "../autodiff/admfem.hpp"
#include "mfem.hpp"
// #include "ascii.hpp"
#include "hpc4mat.hpp"

namespace mfem{

//class BasicNLDiffusionCoefficient;

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

    NLDiffusionIntegrator(
        BasicNLDiffusionCoefficient* mat_,
        Coefficient                  * desfield_)
    {
        mat=mat_;
        desfieldCoeff=desfield_;
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

    BasicNLDiffusionCoefficient * mat = nullptr;
    ParGridFunction             * desfield = nullptr;
    Coefficient             * desfieldCoeff = nullptr;
};

class NLLoadIntegrator:public NonlinearFormIntegrator
{
public:
    NLLoadIntegrator( mfem::ParMesh* pmesh )
    {
        int MaxBdrAttr = pmesh->bdr_attributes.Max();

        ::mfem::Vector Load(MaxBdrAttr);    Load = 0.0;    Load(1) = -0.3; Load(2) = -0.3;
        ::mfem::Coefficient * Coeff = new ::mfem::PWConstCoefficient(Load);
        LFIntegrator = new BoundaryLFIntegrator(*Coeff, 3, 3);
    }

    virtual
    ~NLLoadIntegrator()  
    {
         delete LFIntegrator;
         delete Coeff;
    }


    virtual
    void AssembleFaceVector(
        const FiniteElement &el1,
        const FiniteElement &el2,
        FaceElementTransformations &Tr,
        const Vector &elfun, Vector &elvect)
    {
        LFIntegrator->AssembleRHSElementVect( el1, Tr,  elvect );	
    };

    virtual
    void AssembleFaceGrad(
        const FiniteElement &el1,
        const FiniteElement &el2,
        FaceElementTransformations &Tr,
        const Vector &elfun, DenseMatrix &elmat)
    {
        int dof = el1.GetDof();

        elmat.SetSize(dof);

        elmat = 0.0;
    };

private:

    BoundaryLFIntegrator * LFIntegrator = nullptr;
    ::mfem::Coefficient * Coeff = nullptr;

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
    void SetNewtonSolver(double rtol=1e-8, double atol=1e-12,int miter=10, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-10, double atol=1e-12, int miter=1000)
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

    void AddDesignGF(ParGridFunction* desGF_)
    {
        desfield.push_back(desGF_);
    }

    void AddDesignCoeff(Coefficient* desCoeff_)
    {
        desfieldCoeff.push_back(desCoeff_);
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(fes); sgf.SetFromTrueDofs(sol);}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(fes); agf.SetFromTrueDofs(adj);}

     // Evaluates the compliance
    double evalQoI(int i, double fx, double fy, double fz)
    {
        return 0.0;
    }

    void evalQoIGrad(int i, double fx, double fy, double fz, mfem::Vector& grad)
    {
        
    }

    ParFiniteElementSpace * GetFES()
    {
        return fes;
    }

private:
    mfem::ParMesh* pmesh;

    std::vector<BasicNLDiffusionCoefficient*> materials;
    std::vector<ParGridFunction*>   desfield;
    std::vector<Coefficient*>       desfieldCoeff;

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

    //mfem::HypreBoomerAMG *prec; //preconditioner
    mfem::HypreILU *prec; //preconditioner
    //mfem::CGSolver *ls;  //linear solver
    mfem::GMRESSolver *ls;  //linear solver
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

class EnergyDissipationIntegrator:public NonlinearFormIntegrator
{
public:

    EnergyDissipationIntegrator()
    {};

    void SetPreassure(ParGridFunction* preassure_)
    {
        preassureGF=preassure_;
    };

    void SetNLDiffusionCoeff(BasicNLDiffusionCoefficient * MicroModelCoeff_)
    {
        MicroModelCoeff=MicroModelCoeff_;
    }

    void SetDesingField(ParGridFunction * desfield_)
    {
        desfield=desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * desfield        = nullptr;
};

class EnergyDissipationIntegrator_1:public LinearFormIntegrator
{
public:

    EnergyDissipationIntegrator_1()
    {};

    void SetPreassure(ParGridFunction* preassure_)
    {
        preassureGF=preassure_;
    };

    void SetNLDiffusionCoeff(BasicNLDiffusionCoefficient * MicroModelCoeff_)
    {
        MicroModelCoeff=MicroModelCoeff_;
    }

    void SetDesingField(ParGridFunction * desfield_)
    {
        desfield=desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::ParGridFunction             * preassureGF     = nullptr;
    mfem::BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * desfield        = nullptr;
};

class AdjointPostIntegrator:public LinearFormIntegrator
{
public:

    AdjointPostIntegrator()
    {};

    void SetAdjoint(ParGridFunction* Adjoint_)
    {
        AdjointGF=Adjoint_;
    };

    void SetNLDiffusionCoeff(BasicNLDiffusionCoefficient * MicroModelCoeff_)
    {
        MicroModelCoeff=MicroModelCoeff_;
    }

    void SetDesingField(ParGridFunction * desfield_)
    {
        desfield=desfield_;
    }

    void SetPreassure(ParGridFunction* preassure_)
    {
        preassureGF=preassure_;
    };

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::ParGridFunction             * AdjointGF     = nullptr;
    mfem::BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    mfem::ParGridFunction             * desfield        = nullptr;
    mfem::ParGridFunction             * preassureGF     = nullptr;

};

class MicrostructureVolIntegrator:public NonlinearFormIntegrator
{
public:

    MicrostructureVolIntegrator()
    {};

    void SetDesingField(ParGridFunction * desfield_)
    {
        desfield=desfield_;
    }

    virtual
    double GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun);

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                             const Vector &elfun, DenseMatrix &elmat);

private:
    mfem::ParGridFunction             * desfield        = nullptr;
};

class EnergyDissipationObjective
{
public:
    EnergyDissipationObjective()
    {};

    ~EnergyDissipationObjective()
    { delete nf;};

    void SetNLDiffusionSolver(NLDiffusion* NLDiffSolver_){
        NLDiffSolver=NLDiffSolver_;

        preassureGF= &( NLDiffSolver->GetSolution() );
    };

    void SetPreassure(ParGridFunction* preassure_)
    {
        preassureGF=preassure_;
    };

    void SetDesignFES(ParFiniteElementSpace* fes)
    {
        dfes=fes;
    }

    void SetDesField(ParGridFunction& desfield_)
    {
        desfield=&desfield_;
    }

    void SetNLDiffusionCoeff(BasicNLDiffusionCoefficient * MicroModelCoeff_)
    {
        MicroModelCoeff=MicroModelCoeff_;
    }

    double Eval();

    double Eval(mfem::ParGridFunction& sol);

    void Grad(Vector& grad);

    void Grad(mfem::ParGridFunction& sol, Vector& grad);

private:

    NLDiffusion                 * NLDiffSolver    = nullptr;
    ParFiniteElementSpace       * dfes            = nullptr;
    ParNonlinearForm            * nf              = nullptr;
    EnergyDissipationIntegrator * intgr           = nullptr;
    BasicNLDiffusionCoefficient * MicroModelCoeff = nullptr;
    ParGridFunction             * preassureGF     = nullptr;
    ParGridFunction             * desfield        = nullptr;
};

class VolumeQoI
{
public:
    VolumeQoI()
    {};

    ~VolumeQoI()
    { delete nf;};

    void SetDesignFES(ParFiniteElementSpace* fes)
    {
        dfes=fes;
    }

    void SetDesField(ParGridFunction& desfield_)
    {
        desfield=&desfield_;
    }

    double Eval();

    double Eval(mfem::ParGridFunction& sol);

    void Grad(Vector& grad);

    void Grad(mfem::ParGridFunction& sol, Vector& grad);

private:

    ParFiniteElementSpace       * dfes            = nullptr;
    ParNonlinearForm            * nf              = nullptr;
    MicrostructureVolIntegrator * intgr           = nullptr;
    ParGridFunction             * desfield        = nullptr;
};

    // -----------------------------

    class VelCoefficient : public ::mfem::VectorCoefficient
    {
    public:
        VelCoefficient(
            BasicNLDiffusionCoefficient * MicroModelCoeff,
            ::mfem::ParGridFunction* preassureGF,
            ::mfem::ParGridFunction* desingVarGF)
            : VectorCoefficient(2)

            { 
                MicroModelCoeff_ = MicroModelCoeff;
                preassureGF_ = preassureGF;
                desingVarGF_ = desingVarGF;
            };

        void Eval(
            ::mfem::Vector &V,
            ::mfem::ElementTransformation &T,
            const ::mfem::IntegrationPoint &ip)
            {
                const int dim=T.GetDimension();

                Vector uu; uu.SetSize(dim);
                Vector gradp; gradp.SetSize(dim);

                Vector NNInput(dim+1); NNInput=0.0;

                preassureGF_->GetGradient(T,gradp);

                double DesingThreshold = desingVarGF_->GetValue(T,ip);
                for( int Ik = 0; Ik<dim; Ik ++){ NNInput[Ik] = gradp[Ik];}
                NNInput[dim] = DesingThreshold;

                // fixme add thre
                MicroModelCoeff_->Grad(T,ip,NNInput,uu);

                V.SetSize(dim);
                for( int Ik = 0; Ik<dim; Ik ++){ V[Ik] = uu[Ik];}
            }

    private:

        ::mfem::ParGridFunction * preassureGF_ = nullptr;
        ::mfem::ParGridFunction * desingVarGF_ = nullptr;
        BasicNLDiffusionCoefficient * MicroModelCoeff_ = nullptr;
    }; // end class VelCoefficient





}

#endif
