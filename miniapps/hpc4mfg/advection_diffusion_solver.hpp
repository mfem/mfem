#ifndef ADVECTION_DIFFUSION_SOLVER_HPP
#define ADVECTION_DIFFUSION_SOLVER_HPP

#include "hpc4solvers.hpp"

namespace mfem{

double analytic_T(const Vector &x);

double analytic_solution(const Vector &x);

class Advection_Diffusion_Solver
{
    class RHSAdvCoeff : public mfem::Coefficient
    {
    public:
        RHSAdvCoeff(
            mfem::ParMesh* mesh_,
            mfem::VectorCoefficient* vel,
            mfem::VectorCoefficient* avgTemp,
            double sign = 1.0 ) :
            pmesh_(mesh_),
            vel_(vel),
            avgGradTemp_(avgTemp),
            sign_(sign)
        {

            int dim_=pmesh_->Dimension();

        };

        virtual ~RHSAdvCoeff() {  };

        double Eval(
             mfem::ElementTransformation & T,
             const IntegrationPoint & ip) override;

    private:

    mfem::ParMesh* pmesh_ = nullptr;

    mfem::VectorCoefficient* vel_ = nullptr;

    mfem::VectorCoefficient* avgGradTemp_ = nullptr;

    int dim_ = 0;

    double sign_ = 1.0;
    };

class RHSDiffCoeff : public mfem::VectorCoefficient
{
    public:
        RHSDiffCoeff(
            mfem::ParMesh* mesh_,
            mfem::VectorCoefficient* avgTemp,
            mfem::MatrixCoefficient* Material,
            double sign = 1.0 ) :
            VectorCoefficient(mesh_->Dimension()),
            pmesh_(mesh_),
            avgGradTemp_(avgTemp),
            MaterialCoeff_(Material),
            sign_(sign)
        {

        };

        virtual ~RHSDiffCoeff() {  };

        void Eval(
             mfem::Vector & V,
             mfem::ElementTransformation & T,
             const IntegrationPoint & ip) override;

    private:

    mfem::ParMesh* pmesh_ = nullptr;

    mfem::VectorCoefficient* avgGradTemp_ = nullptr;

    mfem::MatrixCoefficient* MaterialCoeff_; 

    double sign_ = 1.0;
};

public:
    Advection_Diffusion_Solver(mfem::ParMesh* mesh_, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();
        // TODO why not DG space
        //fec = new DG_FECollection(order_, dim, BasisType::GaussLobatto);
        fec = new H1_FECollection(order_,dim);
        fes = new ParFiniteElementSpace(pmesh,fec);
        fes_u = new ParFiniteElementSpace(pmesh,fec,dim);            /// fector space

        sol.SetSize(fes->GetTrueVSize()); sol=0.0;
        rhs.SetSize(fes->GetTrueVSize()); rhs=0.0;

        solgf.SetSpace(fes);

        SetLinearSolver();
    }

    ~Advection_Diffusion_Solver(){
        delete ls;
        delete prec;

        delete fes;
        delete fec;

        delete b;

        delete a;


        for(size_t i=0;i<materials.size();i++){
            delete materials[i];
        }
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

    void SetVelocity( mfem::VectorCoefficient* vel )
    {
        vel_ = vel;
    }

    void SetGradTempMean( mfem::VectorCoefficient* avgGradTemp )
    {
        avgGradTemp_ = avgGradTemp;
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(fes); sgf.SetFromTrueDofs(sol);}

    void Postprocess();

private:
    mfem::ParMesh* pmesh;

    std::vector<MatrixCoefficient*> materials;

    ParBilinearForm *a = nullptr;
    ParLinearForm *b = nullptr;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;

    mfem::VectorCoefficient* vel_ = nullptr;
    mfem::VectorCoefficient* avgGradTemp_ = nullptr;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *fes;
    mfem::ParFiniteElementSpace	  *fes_u;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    mfem::HypreBoomerAMG *prec = nullptr; //preconditioner
    mfem::CGSolver *ls = nullptr;  //linear solver

    // holds DBC in coefficient form
    std::map<int, mfem::Coefficient*> bcc;

    // holds internal DBC
    std::map<int, mfem::ConstantCoefficient> bc;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    // holds internal NBC
    std::map<int, mfem::ConstantCoefficient> nc;

    mfem::Array<int> ess_tdofv;

    ParaViewDataCollection * mPvdc = nullptr;
};




}

#endif
