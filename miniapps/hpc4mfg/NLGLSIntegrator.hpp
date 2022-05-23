#ifndef Stabilization_HPP
#define Stabilization_HPP

#include "mfem.hpp"
#include "hpc4solvers.hpp"

namespace mfem{

double analytic_T(const Vector &x);

double analytic_solution(const Vector &x);


class NLGLSIntegrator :public NonlinearFormIntegrator
{
public:
    NLGLSIntegrator()
    {
        mat=nullptr;
    }

    NLGLSIntegrator(mfem::MatrixCoefficient* mat_, ParGridFunction * GF_)
    {
        mat=mat_;

        U_GF_=GF_;
    }

    NLGLSIntegrator(
        mfem::MatrixCoefficient* mat_,
        ParGridFunction * GF_,
        Coefficient * aCoeff_ )
    {
        mat=mat_;

        U_GF_=GF_;

        Coeff_=aCoeff_;
    }


    void SetMaterial(MatrixCoefficient* mat_)
    {
        mat=mat_;
    }

    void SetVelocity(ParGridFunction* GF_)
    {
        U_GF_=GF_;
    }

    virtual
    ~NLGLSIntegrator(){}

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

    MatrixCoefficient* mat = nullptr;

    ParGridFunction * U_GF_ = nullptr;

    Coefficient * Coeff_ = nullptr;

    bool isSUPG = false;
};

class NLGLS_Solver{
public:
    NLGLS_Solver(mfem::ParMesh* mesh_, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();
        fec=new H1_FECollection(order_,dim);
        fes=new ParFiniteElementSpace(pmesh,fec);
        fes_u=new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(fes->GetTrueVSize()); sol=0.0;
        rhs.SetSize(fes->GetTrueVSize()); rhs=0.0;

        solgf.SetSpace(fes);

        nf=nullptr;
        SetNewtonSolver();
        SetLinearSolver();

        prec=nullptr;
        ls=nullptr;
        ns=nullptr;
    }

    ~NLGLS_Solver(){
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
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve();

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

    /// Add material to the solver. The pointer is owned by the solver.
    void AddMaterial(MatrixCoefficient* nmat)
    {
        materials.push_back(nmat);
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(fes); sgf.SetFromTrueDofs(sol);}

private:
    mfem::ParMesh* pmesh;

    std::vector<MatrixCoefficient*> materials;

    //solution true vector
    mfem::Vector sol;
    //RHS
    mfem::Vector rhs;

    mfem::ParGridFunction solgf;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *fes;
    mfem::ParFiniteElementSpace	  *fes_u;
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
