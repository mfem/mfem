#ifndef MTOP_COEFFICIENTS_HPP
#define MTOP_COEFFICIENTS_HPP

#include "mfem.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <iterator>


#include "../spde/boundary.hpp"
#include "../spde/spde_solver.hpp"
#include "../../linalg/dual.hpp"

namespace mfem {

class RiskMeasures{
public:
    RiskMeasures(const std::vector<double>& samples_)
    {
        samples.resize(samples_.size());
        std::copy(samples_.begin(),samples_.end(),samples.begin());
        prob.resize(0);
        for(size_t i=0;i<samples_.size();i++){
            ind.push_back(i);
        }

        std::sort(ind.begin(),ind.end(),Comp(samples));

        for(size_t i=0;i<samples.size();i++){
            std::cout<<samples[ind[i]]<<" ";
        }
        std::cout<<std::endl;
    }

    RiskMeasures(const std::vector<double>& samples_,const std::vector<double>& prob_)
    {
        samples.resize(samples_.size());
        std::copy(samples_.begin(),samples_.end(),samples.begin());
        prob.resize(prob_.size());
        double sp=0.0;
        for(size_t i=0;i<prob_.size();i++){
            prob[i]=prob_[i];
            sp=sp+prob[i];
        }

        for(size_t i=0;i<prob_.size();i++){
            prob[i]=prob[i]/sp;
            ind.push_back(i);
        }

        std::sort(ind.begin(),ind.end(),Comp(samples));

        /*
        for(size_t i=0;i<samples.size();i++){
            std::cout<<samples[ind[i]]<<" ";
        }
        std::cout<<std::endl;
        for(size_t i=0;i<samples.size();i++){
            std::cout<<prob[ind[i]]<<" ";
        }
        std::cout<<std::endl;
        */

    }


    int VaRIndex(double beta)
    {
        if(prob.size()!=0){
            double cp=0.0;
            int fi=samples.size()-1;
            for(size_t i=0;i<samples.size()-1;i++){
                cp=cp+prob[ind[i]];
                if(cp>(1.0-beta))
                {
                    fi=i;
                    break;
                }
            }
            return ind[fi];

        }else{
            double sp=1.0/(double(samples.size()));
            double cp=0.0;
            int fi=samples.size()-1;
            for(size_t i=0;i<samples.size()-1;i++){
                cp=cp+sp;
                if(cp>(1.0-beta))
                {
                    fi=i;
                    break;
                }
            }
            return ind[fi];
        }
    }

    double VaR(double beta){
        if(prob.size()!=0){
            double cp=0.0;
            int fi=samples.size()-1;
            for(size_t i=0;i<samples.size()-1;i++){
                cp=cp+prob[ind[i]];
                if(cp>(1.0-beta))
                {
                    fi=i;
                    break;
                }
            }
            return samples[ind[fi]];

        }else{
            double sp=1.0/(double(samples.size()));
            int fi=samples.size()-1;
            double cp=0.0;
            for(size_t i=0;i<samples.size()-1;i++){
                cp=cp+sp;
                if(cp>(1.0-beta))
                {
                    fi=i;
                    break;
                }
            }
            return samples[ind[fi]];
        }
    }

    double CVaR(double beta){
        if(fabs(beta-1.0)<std::numeric_limits<double>::epsilon()){
            return Max();
        }

        //find the VaR index
        int vi;
        double cp=0.0;
        if(prob.size()!=0){
            vi=samples.size()-1;
            for(size_t i=0;i<samples.size()-1;i++){
                if((cp+prob[ind[i]])>(1.0-beta))
                {
                    vi=i;
                    break;
                }else{
                    cp=cp+prob[ind[i]];
                }
            }
        }else{
            double sp=1.0/(double(samples.size()));
            vi=samples.size()-1;
            for(size_t i=0;i<samples.size()-1;i++){
                if((cp+sp)>(1.0-beta))
                {
                    vi=i;
                    break;
                }else{
                    cp=cp+sp;}
            }
        }

        double vVaR=samples[ind[vi]];

        double rez=0.0;
        if(prob.size()!=0){
            for(int i=0;i<vi;i++){
                rez=rez+prob[ind[i]]*samples[ind[i]];}
        }else{
            double lp=1.0/samples.size();
            for(int i=0;i<vi;i++){
                rez=rez+lp*samples[ind[i]];}
        }

        rez=rez+(1.0-beta-cp)*vVaR;
        rez=rez/(1.0-beta);

        return rez;
    }


    /// Regularized (smooth) CVaR using the
    /// expression in Beiser2023
    double CVaRe(double beta, double eps=1e-3)
    {
        if(fabs(beta-1.0)<std::numeric_limits<double>::epsilon()){
            return Max();
        }

        double t=0.0;
        //find t
        t=CVaRe_Find_t(beta,eps);

        return CVaRet(beta,eps,t);
    }

    /// For a given t computes regularized CVaR, i.e., CVaRe
    double CVaRet(double beta, double eps, double t){

        std::function<double(double)> rf=[&eps](double y)
        {
            //return y+eps*std::log(1.0+std::exp(-y/eps));
            if(y<0.0){
                return 0.0;
            }else if(y<eps){
                return y*y*y/(eps*eps)-y*y*y*y/(2.0*eps*eps*eps);
            }else{
                return y-eps/2.0;
            }
        };

        std::function<double(double)> rf3=[&eps,&rf](double y)
        {
            return   rf(y+eps/2.0);
        };

        double rez=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                rez=rez+prob[i]*rf3(samples[i]-t);}
        }else{
            double lp=1.0/samples.size();
            for(size_t i=0;i<samples.size();i++){
                rez=rez+lp*rf3(samples[i]-t);}
        }

        return t+rez/(1.0-beta);
    }

    /// For a given t computes the gradient of the
    /// regularized CVaR with respect to t
    double dCVaRe(double beta, double eps, double t){

        std::function<double(double)> rf=[&eps](double y)
        {
            //return 1.0/(1.0+std::exp(-y/eps));
            if(y<0.0){
                return 0.0;
            }else if(y<eps){
                return 3.0*y*y/(eps*eps)-4.0*y*y*y/(2.0*eps*eps*eps);
            }else{
                return 1.0;
            }
        };

        std::function<double(double)> rf3=[&eps,&rf](double y)
        {
            return rf(y+eps/2.0);
        };

        double rez=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                rez=rez+prob[i]*rf3(samples[i]-t);}
        }else{
            double lp=1.0/samples.size();
            for(size_t i=0;i<samples.size();i++){
                rez=rez+lp*rf3(samples[i]-t);}
        }

        rez=1.0-rez/(1.0-beta);
        return rez;
    }


    double CVaRe_Find_t(double beta, double eps,
                        double alpha=1, double rerr=1e-8, int max_it=1000)
    {   if(fabs(beta-1.0)<std::numeric_limits<double>::epsilon()){
            return Max();
        }
        double t=VaR(beta);
        return CVaRe_Find_t(beta,eps,t, alpha, rerr,max_it);
    }

    /// Finds the optimal value of t
    double CVaRe_Find_t(double beta, double eps, double t,
                        double alpha=1, double rerr=1e-8, int max_it=1000)
    {
        if(fabs(beta-1.0)<std::numeric_limits<double>::epsilon()){
            return Max();
        }

        double a=t-1.0*eps;
        double b=t+1.0*eps;
        double tn;
        for(int i=0;i<max_it;i++){
            double dF=dCVaRe(beta,eps,t);
            //std::cout<<"i="<<i<<" dF="<<dF<<" t="<<t<<std::endl;
            tn=a*(b-t)*std::exp(alpha*dF)+b*(t-a);
            tn=tn/((b-t)*std::exp(alpha*dF)+t-a);
            if(fabs(tn-t)<rerr){ t=tn; break;}
            t=tn;
        }
        return t;
    }

    double Mean(){
        double rez=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                rez=rez+prob[i]*samples[i];}
        }else{
            double lp=1.0/samples.size();
            for(size_t i=0;i<samples.size();i++){
                rez=rez+lp*samples[i];}
        }
        return rez;
    }

    double EntRisk(double t){
        double shift=Max();

        double lt=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                lt=lt+prob[i]*std::exp((samples[i]-shift)/t);
            }
        }else{
            double sp=1.0/double(samples.size());
            for(size_t i=0;i<samples.size();i++){
                lt=lt+sp*std::exp((samples[i]-shift)/t);
            }
        }

        lt=std::log(lt);
        return shift+t*lt;
    }

    double Max()
    {
        return samples[ind[0]];
    }

    double Min()
    {
        return samples[ind[samples.size()-1]];
    }

    //eval EVaR
    double EVaR(double beta, double alpha=1.0, double eps=1e-4, int max_it=1000){


        if(fabs(beta)<std::numeric_limits<double>::epsilon()){
            return Mean();
        }

        if(fabs(beta-1.0)<std::numeric_limits<double>::epsilon()){
            return samples[ind[0]];
        }

        //find t
        double t=EVaR_Find_t(beta,alpha,eps,max_it);
        return EVaRt(beta,t);
    }

    // For given beta and t evaluates EVaR
    template<typename tfloat>
    tfloat EVaRt(tfloat beta, tfloat t){

        double shift=Max();

        tfloat lt; lt=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                lt=lt+prob[i]*exp((samples[i]-shift)/t);
            }
        }else{
            double sp=1.0/double(samples.size());
            for(size_t i=0;i<samples.size();i++){
                lt=lt+sp*exp((samples[i]-shift)/t);
            }
        }

        lt=lt/(1.0-beta);
        lt=log(lt);
        return shift+t*lt;
    }

    double ADEVaRt(double beta, double t)
    {
        typedef internal::dual<double,double> ADType;
        typedef internal::dual<ADType,ADType> SDType;
        SDType abeta;
        SDType adt;
        SDType lt;

        abeta.value.value=beta;
        abeta.value.gradient=0.0;
        abeta.gradient.value=0.0;
        abeta.gradient.gradient=0.0;

        lt.value.value=t;
        lt.value.gradient=1.0;
        lt.gradient.value=1.0;
        lt.gradient.gradient=1.0;


        bool flag=true;
        while(flag){
            SDType rez=EVaRt(abeta,lt);
            double dt=-rez.value.gradient/rez.gradient.gradient;
            lt.value.value+=dt;
            std::cout<<" dt="<<dt<<" xn="<<lt.value.value<<" rp="<<rez.value.value<<std::endl;
            if(fabs(dt)<1e-4){break;}
        }



        adt.value.value=t;
        adt.value.gradient=1.0;
        adt.gradient.value=1.0;
        adt.gradient.gradient=1.0;


        SDType rez=EVaRt(abeta,adt);

        std::cout<<" rez.v="<<rez.value.value<<" rez.d="<<rez.value.gradient;
        std::cout<<" rez.g="<<rez.gradient.value<<" rez.s="<<rez.gradient.gradient<<std::endl;

        return rez.value.gradient;

    }

    double EVaR_Find_t(double beta, double alpha=1.0, double eps=1e-4, int max_it=1000)
    {
        double tmax=fabs(Mean());
        std::vector<std::tuple<double,double>> dat;
        std::vector<std::tuple<double,double>> tad;
        std::vector<double> tt;

        double dt=tmax/10.0;
        double t=0.0;
        double e;

        for(int i=0;i<12;i++){
            t=t+dt; e=EVaRt(beta,t);
            dat.push_back(std::make_tuple(e,t));
        }

        std::sort(dat.begin(),dat.end());
        //for(int i=0;i<12;i++){ std::cout<<std::get<0>(dat[i])<<" ";} std::cout<<std::endl;

        for(int i=0;i<4;i++){ tad.push_back(dat[i]);}
        dat.clear();

        t=0.25*std::get<1>(tad[0])+0.75*std::get<1>(tad[1]);
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));

        t=0.5*std::get<1>(tad[0])+0.5*std::get<1>(tad[1]);
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));

        t=0.75*std::get<1>(tad[0])+0.25*std::get<1>(tad[1]);
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));

        t=0.25*std::get<1>(tad[0])+0.75*std::get<1>(tad[2]);
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));

        t=0.5*(std::get<1>(tad[0])+std::get<1>(tad[2]));
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));

        t=0.75*std::get<1>(tad[0])+0.25*std::get<1>(tad[2]);
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));


        for(int i=0;i<6;i++){tt.push_back(std::get<1>(tad[i]));}
        std::sort(tt.begin(),tt.end());
        dt=tt[0]/3.0;
        e=EVaRt(beta,dt);
        tad.push_back(std::make_tuple(e,dt));
        t=tt[6]+dt;
        e=EVaRt(beta,t);
        tad.push_back(std::make_tuple(e,t));
        std::sort(tad.begin(),tad.end());

        //for(int i=0;i<tad.size();i++){ std::cout<<std::get<0>(tad[i])<<" ";} std::cout<<std::endl;
        //for(int i=0;i<tad.size();i++){ std::cout<<std::get<1>(tad[i])<<" ";} std::cout<<std::endl;



        double tmin=std::get<1>(tad[0]);

        return EVaR_Find_t(beta,tmin,alpha,eps,max_it);
    }

    double EVaR_Find_t(double beta, double t, double alpha=1.0, double eps=1e-4, int max_it=1000)
    {
        //find t
        double inc=0.0;
        int iter=0;
        double dF;
        while(fabs(inc-1.0)>eps)
        {
            dF=dEVaR(beta,t);
            //std::cout<<" i="<<iter<<" dF="<<dF<<" inc="<<exp(-alpha*dF)<<" t="<<t<<" e="<<EVaRt(beta,t*exp(-alpha*dF))<<std::endl;
            inc=exp(-alpha*dF);
            if(inc>1.1){inc=1.1;}
            if(inc<0.9){inc=0.9;}
            t=t*inc;
            iter++;
            if(iter==max_it){
                //MFEM_WARNING("Maximum number of iterations in EVaR_Find_t has been reached!!!")
                break;
            }
            if(t<1e-8){
                break;
            }
        }
        //std::cout<<std::endl<<"iter="<<iter<<" t="<<t<<std::endl;

        return t;
    }

    void Test_EVaR_Grad()
    {
        double beta=0.9;
        double t=0.4;
        double dt=1.0;
        double tt;
        double EReF=EVaRt(beta,t);
        double EVaR;
        double dF=dEVaR(beta,t);

        double CReFe=CVaRet(beta,1e-3,t);
        double CVaRe;
        double dC=dCVaRe(beta,1e-3,t);

        for(int i=0;i<10;i++){
            tt=t+dt;
            EVaR=EVaRt(beta,tt);
            std::cout<<" FD="<<(EVaR-EReF)/dt<<" DF="
                    <<dF<<" err="<<fabs(dF-(EVaR-EReF)/dt)<<std::endl;

            CVaRe=CVaRet(beta,1e-3,tt);
            std::cout<<" FD="<<(CVaRe-CReFe)/dt<<" DF="
                    <<dC<<" err="<<fabs(dC-(CVaRe-CReFe)/dt)<<std::endl;


            dt=dt/10.0;
        }

    }

    double dEVaR(double beta, double t){
        double shift=Max();
        double rez=0.0;
        double lt=0.0;
        double nt=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                lt=lt+prob[i]*std::exp((samples[i]-shift)/t);
                nt=nt+prob[i]*samples[i]*std::exp((samples[i]-shift)/t);
            }
        }else{
           double sp=1.0/double(samples.size());
           for(size_t i=0;i<samples.size();i++){
               lt=lt+sp*std::exp((samples[i]-shift)/t);
               nt=nt+sp*samples[i]*std::exp((samples[i]-shift)/t);
           }
        }

        //std::cout<<" t1="<<shift/t<<" t2="<<std::log(lt/(1.0-beta))<<" t3="<<nt/(lt*t)<<std::endl;
        rez=shift/t+std::log(lt/(1.0-beta))-nt/(lt*t);
        return rez;
    }

    double STD(){
        double var=0.0;
        double rez=0.0;
        if(prob.size()!=0){
            for(size_t i=0;i<samples.size();i++){
                rez=rez+prob[i]*samples[i];
                var=var+prob[i]*samples[i]*samples[i];
            }
        }else{
            double lp=1.0/samples.size();
            for(size_t i=0;i<samples.size();i++){
                rez=rez+lp*samples[i];
                var=var+lp*samples[i]*samples[i];
            }
        }
        return std::sqrt(var-rez*rez);
    }



private:

    std::vector<int> ind;
    std::vector<double> samples;
    std::vector<double> prob;

    struct Comp{
        Comp(std::vector<double>& ss_)
        {
            ss=&ss_;
        }

        bool operator()(int i1,int i2){
            return (*ss)[i1]>(*ss)[i2];
        }

    private:
        std::vector<double>* ss;
    };


};





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


class ThresholdCoefficient:public Coefficient
{
public:
    ThresholdCoefficient(Coefficient* gf_, double t_=1.5, double a_=0.0, double b_=1.0):t(t_),a(a_),b(b_)
    {
        gf=gf_;
    }

    void SetGaussianCoeff(Coefficient* gf_){
        gf=gf_;
    }

    void SetThresholdValues(double a_,double b_)
    {
        a=a_;
        b=b_;
    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip){
        double val=gf->Eval(T,ip);
        if(val>t){ return a;}
        else{ return b;}
    }

private:
    double t;
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

        angle_x=0.0;
        angle_y=0.0;
        angle_z=0.0;

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

    void SetZeroDirichletBC(int i)
    {
        dbc.insert(i);
    }

    void SetCorrelationLen(double lx_, double ly_, double lz_){
        lx=lx_;
        ly=ly_;
        lz=lz_;
        delete solver; solver=nullptr;
    }

    void SetCorrelationLen(double lx_){
        lx=lx_;
        ly=lx_;
        lz=lx_;
        delete solver; solver=nullptr;
    }

    void SetRotationAngles(double angle_x_, double angle_y_, double angle_z_){
        angle_x=angle_x_;
        angle_y=angle_y_;
        angle_z=angle_z_;
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

    void Sample(int seed=std::numeric_limits<int>::max()){
        if(solver==nullptr){
            if(dbc.size()!=0){
                for(auto it=dbc.begin();it!=dbc.end();it++){
                    bc.AddHomogeneousBoundaryCondition(*it,spde::BoundaryType::kDirichlet);
                }
            }
            solver=new spde::SPDESolver(nu,bc,fes,pmesh->GetComm(),lx,ly,lz,angle_x, angle_y, angle_z);
        }
        solver->GenerateRandomField(*rf,seed);
        gfc.SetGridFunction(rf);
    }

private:
    double lx,ly,lz;
    double angle_x,angle_y,angle_z;
    double nu;
    double scale;
    ParMesh* pmesh;
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fes;
    spde::SPDESolver* solver;
    ParGridFunction* rf;
    spde::Boundary bc;
    GridFunctionCoefficient gfc;
    std::set<int> dbc;//Dirichlet BC
};

#endif

}

#endif
