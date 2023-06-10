#ifndef MTOP_COEFFICIENTS_HPP
#define MTOP_COEFFICIENTS_HPP

#include "mfem.hpp"
#include <random>

#include "../spde/boundary.hpp"
#include "../spde/spde_solver.hpp"

namespace mfem {


/// Generates a coefficient with value zero or one.
/// The value is zero if the point is within a ball
/// with radius r and randomly generated center within
/// the boundig box of the mesh. The value is one otherwise.
/// If the mesh is not rectangular one should check
/// if the generated sample is within the mesh.
class RandShootingCoefficient:public Coefficient
{
public:
    RandShootingCoefficient(Mesh* mesh_, double r):udist(0,1)
    {
        mesh=mesh_;
        mesh->GetBoundingBox(cmin,cmax);
        radius=r;

        center.SetSize(cmin.Size());
        center=0.0;

        tmpv.SetSize(cmin.Size());
        tmpv=0.0;

        Sample();
    }

    virtual
    ~RandShootingCoefficient()
    {

    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip);

    /// Generates a new random center for the ball.
    void Sample();

private:

    double radius;
    Vector center;
    Vector cmin;
    Vector cmax;

    Vector tmpv;

    Mesh* mesh;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist;

};

class LognormalDistributionCoefficient:public Coefficient
{
public:
    LognormalDistributionCoefficient(Coefficient* gf_,double mu_=0.0, double ss_=1.0):mu(mu_),ss(ss_)
    {
        gf=gf_;
        scale=1.0;
    }

    void SetGaussianCoeff(Coefficient* gf_){
        gf=gf_;
    }

    void SetScale(double sc_){
        scale=sc_;
    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip){
        double val=gf->Eval(T,ip);
        return scale*std::exp(mu+ss*val);
    }

private:
    double mu;
    double ss;
    Coefficient* gf;
    double scale;
};


class UniformDistributionCoefficient:public Coefficient
{
public:
    UniformDistributionCoefficient(Coefficient* gf_, double a_=0.0, double b_=1.0):a(a_),b(b_)
    {
       gf=gf_;
    }

    void SetGaussianCoeff(Coefficient* gf_){
        gf=gf_;
    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip){
        double val=gf->Eval(T,ip);
        return a+(b-a)*(1.0+std::erf(-val/std::sqrt(2.0)))/2.0;
    }

private:
    double a;
    double b;
    Coefficient* gf;

};

#ifdef MFEM_USE_MPI

class RandFieldCoefficient:public Coefficient
{
public:
    RandFieldCoefficient(ParMesh* mesh_, int order){
        pmesh=mesh_;
        fec=new H1_FECollection(order,mesh_->Dimension());
        fes=new ParFiniteElementSpace(pmesh,fec,1);

        lx=1.0;
        ly=1.0;
        lz=1.0;

        solver=nullptr;
        rf=new ParGridFunction(fes); (*rf)=0.0;
        gfc.SetGridFunction(rf);

        scale=1.0;

    }

    ~RandFieldCoefficient(){
        delete solver;
        delete rf;
        delete fes;
        delete fec;
    }

    void SetScale(double scale_=1.0){
        scale=scale_;
    }

    void SetCorrelationLen(double l_){
        lx=l_;
        ly=l_;
        lz=l_;
        delete solver; solver=nullptr;
    }

    void SetMaternParameter(double nu_){
        nu=nu_;
        delete solver; solver=nullptr;
    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip)
    {
        return scale*gfc.Eval(T,ip);
    }

    /// Generates a new random center for the ball.
    void Sample(int seed=std::numeric_limits<int>::max()){
        if(solver==nullptr){
            solver=new spde::SPDESolver(nu,bc,fes,pmesh->GetComm(),lx,ly,lz);
        }
        solver->GenerateRandomField(*rf,seed);
        gfc.SetGridFunction(rf);
    }

private:
    double lx,ly,lz;
    double nu;
    double scale;
    ParMesh* pmesh;
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fes;
    spde::SPDESolver* solver;
    ParGridFunction* rf;
    spde::Boundary bc;
    GridFunctionCoefficient gfc;
};

#endif

}

#endif
