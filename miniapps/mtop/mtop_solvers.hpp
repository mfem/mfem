#ifndef MTOP_SOLVERS_HPP
#define MTOP_SOLVERS_HPP

#include "mfem.hpp"
#include "rand_eigensolver.hpp"

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


    /// operator A is considered to be symmetric
    class LocProductOperator : public Operator
    {
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
            y=0.0;
            A->Mult(z, y);


            /*
            y=0.0;
            A->Mult(x, y);
            */

            /*
            B->Mult(x,y);
            */
        }

        void MultTranspose(const Vector &x, Vector &y) const override
        {

            z=0.0;
            A->Mult(x, z);
            B->MultTranspose(z, y);

            /*
            y=0.0;
            A->Mult(x, y);
            */
            /*
            B->Mult(x,y);
            */
        }

        void MultB(const Vector &x, Vector &y)
        {
            B->Mult(x, y);
        }

        void MultA(const Vector &x, Vector &y)
        {
            y=0.0;
            A->Mult(x,y);
        }

        virtual ~LocProductOperator(){

        }
    };


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
#endif // MTOP_SOLVERS_HPP
