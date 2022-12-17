#ifndef ELPLAST_HPP
#define ELPLAST_HPP

#include "mfem.hpp"
#include "coefficients.hpp"

namespace mfem{




class NLElPlastIntegrator:public BlockNonlinearFormIntegrator
{
public:
    NLElPlastIntegrator(double E_=1, double nu_=0.2, double ss_y_=1.0, double ll_=0.1)
    {
        lE=new ConstantCoefficient(E_);
        lnu=new ConstantCoefficient(nu_);
        lss_y=new ConstantCoefficient(ss_y_);
        lll=new ConstantCoefficient(ll_);

        E=lE;
        nu=lnu;
        ss_y=lss_y;
        ll=lll;

        eep=nullptr;
        kappa=nullptr;

        force=nullptr;
        flag_update=false;
    }

    virtual ~NLElPlastIntegrator()
    {
        delete lE;
        delete lnu;
        delete lss_y;
        delete lll;
    }

    void SetPlasticStrains(mfem::QuadratureFunction& eep_,
                           mfem::QuadratureFunction& kappa_)
    {
        eep=&eep_;
        kappa=&kappa_;
    }

    void SetForce(VectorCoefficient& vc)
    {
        force=&vc;
    }

    void SetUpdateFlag(bool fl)
    {
        flag_update=fl;
    }

    virtual
    double GetElementEnergy(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun)
    {
        return 0.0;
    }


    virtual
    void AssembleElementVector(const Array<const FiniteElement *> &el,
                               ElementTransformation &Tr,
                               const Array<const Vector *> &elfun,
                               const Array<Vector *> &elvec);

    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array2D<DenseMatrix *> &elmats);

private:

    mfem::Coefficient* lE;
    mfem::Coefficient* lnu;
    mfem::Coefficient* lss_y;
    mfem::Coefficient* lll;


    mfem::Coefficient* E;
    mfem::Coefficient* nu;
    mfem::Coefficient* ss_y;
    mfem::Coefficient* ll;

    mfem::VectorCoefficient* force;

    mfem::QuadratureFunction* eep;
    mfem::QuadratureFunction* kappa;

    mfem::IsoElastMat mat;
    mfem::J2YieldFunction yf;

    bool flag_update;

};

class NLElPlastIntegratorS:public NonlinearFormIntegrator
{
public:

    NLElPlastIntegratorS(double E_=1, double nu_=0.2, double ss_y_=1.0, double ll_=0.1)
    {
        lE=new ConstantCoefficient(E_);
        lnu=new ConstantCoefficient(nu_);
        lss_y=new ConstantCoefficient(ss_y_);
        lll=new ConstantCoefficient(ll_);

        E=lE;
        nu=lnu;
        ss_y=lss_y;
        ll=lll;

        eep=nullptr;
        kappa=nullptr;
    }

    virtual ~NLElPlastIntegratorS()
    {
        delete lE;
        delete lnu;
        delete lss_y;
        delete lll;
    }

    void SetPlasticStrains(mfem::QuadratureFunction& eep_,
                           mfem::QuadratureFunction& kappa_)
    {
        eep=&eep_;
        kappa=&kappa_;
    }

    void SetFilteredPlasticStrain(GridFunction& epf)
    {
        eef=&epf;
    }

    virtual
    double GetElementEnrgy(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
    {
        return 0.0;
    }

    virtual
    void AssembleElementVector (const FiniteElement &el,
                                ElementTransformation &Tr,
                                const Vector &elfun, Vector &elvect);

    virtual
    void AssembleElementGrad (const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, DenseMatrix &elmat);


private:
    mfem::Coefficient* lE;
    mfem::Coefficient* lnu;
    mfem::Coefficient* lss_y;
    mfem::Coefficient* lll;


    mfem::Coefficient* E;
    mfem::Coefficient* nu;
    mfem::Coefficient* ss_y;
    mfem::Coefficient* ll;

    mfem::QuadratureFunction* eep;
    mfem::QuadratureFunction* kappa;
    mfem::GridFunction* eef;

    mfem::IsoElastMat mat;
    mfem::J2YieldFunction yf;
};



class ElPlastSolver
{
public:
    ElPlastSolver(mfem::ParMesh* mesh_,int vorder=1, int forder=2);

    ~ElPlastSolver();

    /// Set the Newton Solver
    void SetNewtonSolver(double rtol=1e-7, double atol=1e-12,int miter=1000, int prt_level=1);

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1000);

    /// Solves the forward problem.
    void FSolve();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::Vector& rhs);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, double val);

    /// Adds displacement BC in direction 0(x),1(y),2(z), or 4(all).
    void AddDispBC(int id, int dir, mfem::Coefficient& val);

    /// Adds displacement BC specified by the vector coefficient val.
    void AddDispBC(int id, mfem::VectorCoefficient& val);

    /// Adds vol force
    void AddVolForce(int id, double fx, double fy, double fz);

    /// Adds vol force
    void AddVolForce(int id, mfem::VectorCoefficient& ff);

    /// Returns the displacements.
    mfem::ParGridFunction& GetDisplacements()
    {
        fdisp.SetFromTrueDofs(sol.GetBlock(0));
        return fdisp;
    }

    /// Returns the adjoint displacements.
    mfem::ParGridFunction& GetADisplacements()
    {
        adisp.SetFromTrueDofs(adj.GetBlock(0));
        return adisp;
    }

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(vfes); sgf.SetFromTrueDofs(sol.GetBlock(0));}

    void GetAdj(ParGridFunction& agf){
        agf.SetSpace(vfes); agf.SetFromTrueDofs(adj.GetBlock(0));}
private:

    double current_time;

    mfem::ParMesh* pmesh;

    //solution vector
    mfem::BlockVector sol;
    //adjoint vector
    mfem::BlockVector adj;
    //RHS
    mfem::BlockVector rhs;

    // localy defined volumetric forces
    std::map<int, mfem::VectorConstantCoefficient*> lvforce;
    // globaly defined volumetric forces
    std::map<int, mfem::VectorCoefficient*> volforce;

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;
    std::map<int, mfem::VectorCoefficient*> bcca;

    // holds the displacement contrained DOFs
    mfem::Array<int> ess_tdofv;

    //forward solution
    mfem::ParGridFunction fdisp;
    //adjoint solution
    mfem::ParGridFunction adisp;

    //total strains
    mfem::QuadratureFunction eee;
    //plastic strains
    mfem::QuadratureFunction eep;
    //accumulated plastic strain
    mfem::QuadratureFunction kappa;

    //Newton solver parameters
    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    mfem::ParBlockNonlinearForm *nf;
    mfem::ParFiniteElementSpace* vfes; //displacements fes
    mfem::ParFiniteElementSpace* ffes; //filter fes

    mfem::FiniteElementCollection* vfec;
    mfem::FiniteElementCollection* ffec;

    mfem::QuadratureSpace* qfes;
};

}

#endif
