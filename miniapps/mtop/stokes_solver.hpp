#ifndef STOKES_SOLVER_HPP
#define STOKES_SOLVER_HPP

#include "mfem.hpp"

class StokesOperator:public mfem::Operator
{
public:
    StokesOperator(mfem::ParMesh* mesh_,int vorder=2);

    virtual ~StokesOperator();

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

    /// Adds velocity BC in direction 0(x),1(y),2(z), or 4(all).
    void AddVelocityBC(int id, int dir, double val);

    /// Adds velocity BC in direction 0(x),1(y),2(z), or 4(all).
    void AddVelocityBC(int id, int dir, mfem::Coefficient& val);

    /// Clear all velocity BC
    void DelVelocityBC();

    /// Set the values of the volumetric force.
    void SetVolForce(double fx,double fy, double fz=0.0);

    /// Add surface load
    void AddSurfLoad(int id, double fx,double fy, double fz=0.0)
    {
        /*
        mfem::Vector vec; vec.SetSize(pmesh->SpaceDimension());
        vec[0]=fx;
        vec[1]=fy;
        if(pmesh->SpaceDimension()==3){vec[2]=fz;}
        mfem::VectorConstantCoefficient* vc=new mfem::VectorConstantCoefficient(vec);
        if(load_coeff.find(id)!=load_coeff.end()){ delete load_coeff[id];}
        load_coeff[id]=vc;
        */
    }

    /// Add surface load
    void AddSurfLoad(int id, mfem::VectorCoefficient& ff)
    {
        /*
        surf_loads[id]=&ff;
        */
    }

    /// Associates coefficient to the volumetric force.
    void SetVolForce(mfem::VectorCoefficient& ff);


    /// Sets BC dofs, bilinear form, preconditioner and solver.
    /// Should be called before calling Mult of MultTranspose
    virtual void Assemble();

    /// Forward solve with given RHS. x is the RHS vector. The BC are set to zero.
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

    /// Adjoint solve with given RHS. x is the RHS vector. The BC are set to zero.
    virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

    /// Set material
    void SetViscosoty(mfem::Coefficient& mu_)
    {
        mu=&mu_;

        delete af; af=nullptr;
    }

    void SetBrinkman(mfem::Coefficient& alpha_)
    {
        brink=&alpha_;

        delete af; af=nullptr;
    }

    /// Returns the velocity field
    mfem::ParGridFunction& GetVelocity()
    {
        fvelo.SetFromTrueDofs(sol.GetBlock(0));
        return fvelo;
    }

    /// Return the pressure field
    mfem::ParGridFunction& GetPressure()
    {
        fpres.SetFromTrueDofs(sol.GetBlock(1));
        return fpres;
    }

    void GetVelocity(mfem::ParGridFunction& v)
    {
        v.SetSpace(vfes);
        v.SetFromTrueDofs(sol.GetBlock(0));
    }

    void GetPressure(mfem::ParGridFunction& p)
    {
        p.SetSpace(pfes);
        p.SetFromTrueDofs(sol.GetBlock(1));
    }


    class InverseCoeff:public mfem::Coefficient
    {
    public:
        InverseCoeff(mfem::Coefficient* co)
        {
            cc=co;
        }

        void SetCoeff(mfem::Coefficient* co)
        {
            cc=co;
        }


        virtual mfem::real_t
        Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
        {
            mfem::real_t vv=cc->Eval(T,ip);
            return 2.0/(1.0*vv);
            //return vv;
        }

    private:
        mfem::Coefficient* cc;
    };

protected:
    mfem::ParMesh* pmesh;

    mfem::Array<int> block_offsets; // number of variables + 1
    mfem::Array<int> block_true_offsets;

    //block vectors
    mutable mfem::BlockVector sol;
    mutable mfem::BlockVector adj;
    mutable mfem::BlockVector rhs;

    mfem::ParGridFunction fvelo;
    mfem::ParGridFunction fpres;

    mfem::ParGridFunction avelo;
    mfem::ParGridFunction apres;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    //finite element space for velocity
    mfem::ParFiniteElementSpace* vfes;
    //finite element collection for velocity
    mfem::FiniteElementCollection* vfec;

    //finite element space for pressure
    mfem::ParFiniteElementSpace* pfes;
    //finite element collection for pressure
    mfem::FiniteElementCollection* pfec;


    mfem::ConstantCoefficient zeroc;
    mfem::ConstantCoefficient onec;
    mfem::Coefficient* mu;
    mfem::Coefficient* brink; //Brinkman penalization


    mfem::ParBilinearForm* af;
    mfem::ParMixedBilinearForm* bf;
    std::unique_ptr<mfem::ParBilinearForm> mf;

    mfem::ParLinearForm* lf; //forces
    mfem::ParLinearForm* pf; //pressure rhs

    // boundary conditions for x,y, and z directions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    // holds BC in coefficient form
    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

    // holds the velocity constrained DOFs
    mfem::Array<int> ess_tdofv;

    // holds the pressure constrained DOFs
    mfem::Array<int> ess_tdofp;

    std::unique_ptr<mfem::HypreParMatrix> A;
    std::unique_ptr<mfem::HypreParMatrix> Ae;
    std::unique_ptr<mfem::HypreParMatrix> B;
    std::unique_ptr<mfem::HypreParMatrix> Be;
    std::unique_ptr<mfem::HypreParMatrix> D;
    std::unique_ptr<mfem::HypreParMatrix> De;
    std::unique_ptr<mfem::HypreParMatrix> M;
    std::unique_ptr<mfem::HypreParMatrix> Me;

    std::unique_ptr<InverseCoeff> ivisc;

    mfem::BlockOperator* bop; //tangent operator
    mfem::Solver* pop; //preconditioner

    mfem::IterativeSolver *ls;

    void SetEssTDofs(mfem::Vector& bsol, mfem::Array<int>& ess_dofs);


    void AssemblePrec1();
    void AssemblePrec2();
    void AssemblePrec3();
    void AssemblePrec4();
    void AssemblePrec5();
    std::unique_ptr<mfem::HypreBoomerAMG> prec1;
    std::unique_ptr<mfem::HypreBoomerAMG> prec2;


};



#endif // STOKESSOLVER_HPP
