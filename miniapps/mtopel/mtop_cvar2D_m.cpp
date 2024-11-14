#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"

#include "mtop_coefficients.hpp"
#include "mtop_solvers.hpp"

#include <bitset>

namespace adsampl {

double inv_sigmoid(double x, double p=1.0, double a=0.0)
{
    double tol = 1e-12;
    double c=p/(1.0-a);
    x = std::min(std::max(tol,x), c-tol);
    return std::log(x/(c-x));
}

/// @brief Sigmoid function
double sigmoid(double x,double p=1.0, double a=0.0)
{
    double s=p/(1.0-a);
    if (x >= 0)
    {
        return s/(1.0+std::exp(-x));
    }
    else
    {
        return s*std::exp(x)/(1.0+std::exp(x));
    }
}

/// @brief Derivative of sigmoid function
double der_sigmoid(double x,double p=1.0, double a=0.0)
{
    double s=p/(1.0-a);
    double tmp = sigmoid(-x);
    return s*(tmp - std::pow(tmp,2));
}

// ind should consists of unique indices
// f should be the size of ind
double Find_t(const std::vector<double>& p,
              const std::vector<double>& q,
              double alpha, double gamma,
              const std::vector<double>& f,
              const std::vector<int>& ind,
              double tol=1e-12,
              int max_it=100)
{

    if(ind.size()==0){
        return 0.0;
    }


    double cval=-1.0;
    {
        std::vector<bool> pv; pv.resize(p.size());
        for(size_t i=0;i<p.size();i++){
            pv[i]=true;
        }
        for(size_t i=0;i<ind.size();i++){
            pv[ind[i]]=false;
        }
        for(size_t i=0;i<p.size();i++){
            if(pv[i]){	cval=cval+q[i]; }
        }
    }

    for(size_t i=0;i<p.size();i++){
        std::cout<<" "<<p[i];
    }
    std::cout<<std::endl;
    for(size_t i=0;i<q.size();i++){
        std::cout<<" "<<q[i];
    }
    std::cout<<std::endl;
    for(size_t i=0;i<f.size();i++){
        std::cout<<" "<<f[i];
    }
    std::cout<<std::endl;


    std::cout<<"cval="<<cval<<std::endl;


    std::vector<double> g; g.resize(f.size());
    {
        for(size_t i=0;i<ind.size();i++){
            g[i]=inv_sigmoid(q[ind[i]],p[ind[i]],alpha)+gamma*f[i];
        }
    }

    for(size_t i=0;i<g.size();i++){
        std::cout<<" "<<g[i];
    }
    std::cout<<std::endl;

    bool flag=false; //iteration flag
    int iter=0;

    double ff;
    double df;
    double tt=0.0;
    double dc=0.0;

    for(int k=0;k<max_it;k++)
    {
        iter++;

        ff=cval;
        df=0.0;

        for(size_t i=0;i<ind.size();i++){
            ff=ff+sigmoid(g[i]-tt,p[ind[i]],alpha);
            df=df-der_sigmoid(g[i]-tt,p[ind[i]],alpha);
        }

        std::cout<<"tt="<<tt<<" ff="<<ff<<" df="<<df<<" dc="<<dc<<std::endl;

        if(fabs(df)<tol){break;}
        dc=-ff/df;
        tt=tt+dc;
        std::cout<<"tt="<<tt<<" ff="<<ff<<" df="<<df<<" dc="<<dc<<std::endl;
        if(fabs(dc)<tol){flag=true; break;}
    }

    if(!flag){
        mfem::mfem_warning("Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }

    return tt;
}

// ind should consists of unique indices
// f should be the size of ind
double Find_t_bisection(const std::vector<double>& p,
                        const std::vector<double>& q,
                        double alpha, double gamma,
                        const std::vector<double>& f,
                        const std::vector<int>& ind,
                        double tol=1e-12,
                        int max_it=100)
{

    if(ind.size()==0){
        return 0.0;
    }

    double cval=-1.0;
    {
        std::vector<bool> pv; pv.resize(p.size());
        for(size_t i=0;i<p.size();i++){
            pv[i]=true;
        }
        for(size_t i=0;i<ind.size();i++){
            pv[ind[i]]=false;
        }
        for(size_t i=0;i<p.size();i++){
            if(pv[i]){	cval=cval+q[i]; }
        }
    }

    /*
    for(size_t i=0;i<p.size();i++){
        std::cout<<" "<<p[i];
    }
    std::cout<<std::endl;
    for(size_t i=0;i<q.size();i++){
        std::cout<<" "<<q[i];
    }
    std::cout<<std::endl;
    for(size_t i=0;i<f.size();i++){
        std::cout<<" "<<f[i];
    }
    std::cout<<std::endl;
    for(size_t i=0;i<f.size();i++){
        std::cout<<" "<<ind[i];
    }
    std::cout<<std::endl;
    */


    std::cout<<"cval="<<cval<<std::endl;

    std::vector<double> g; g.resize(f.size());
    {
        for(size_t i=0;i<ind.size();i++){
            g[i]=inv_sigmoid(q[ind[i]],p[ind[i]],alpha)+gamma*f[i];
        }
    }

    /*
    for(size_t i=0;i<g.size();i++){
        std::cout<<" "<<g[i];
    }
    std::cout<<std::endl;
    */

    double fmin=gamma*f[0];
    double fmax=gamma*f[0];

    for(size_t i=1;i<ind.size();i++)
    {
        if(fmin>gamma*f[i]){fmin=gamma*f[i];}
        if(fmax<gamma*f[i]){fmax=gamma*f[i];}
    }

    double tmin=fmax+fabs(fmax)*0.01;
    double tmax=fmin-fabs(fmin)*0.01;

    double umin=cval;
    double umax=cval;

    fmin=cval;
    fmax=cval;
    for(size_t i=0;i<ind.size();i++){
        fmin=fmin+sigmoid(g[i]-tmin,p[ind[i]],alpha);
        fmax=fmax+sigmoid(g[i]-tmax,p[ind[i]],alpha);

        umin=umin+sigmoid(inv_sigmoid(q[ind[i]],p[ind[i]],alpha),p[ind[i]],alpha);
        umax=umax+q[ind[i]];
    }

    std::cout<<"tmin="<<tmin<<" fmin="<<fmin<<" gma="<<gamma<<" u="<<umax<<std::endl;
    std::cout<<"tmax="<<tmax<<" fmax="<<fmax<<" gma="<<gamma<<" u="<<umax<<std::endl;

    if(fmin>fmax){
        std::swap(tmin,tmax);
        std::swap(fmin,fmax);
    }

    int iter=0;

    double ff;
    double tt;

    std::cout<<"fmin="<<fmin<<" fmax="<<fmax<<std::endl;


    bool flag=false; //iteration flag

    for(int k=0;k<max_it;k++)
    {
        iter++;

        ff=cval;
        tt=(tmin+tmax)/2.0;

        for(size_t i=0;i<ind.size();i++){
            ff=ff+sigmoid(g[i]-tt,p[ind[i]],alpha);
        }

        //std::cout<<"tt="<<tt<<" ff="<<ff<<std::endl;

        if(ff<0.0){fmin=ff; tmin=tt;}
        else{fmax=ff; tmax=tt;}

        if((fmax-fmin)<tol){flag=true; break;}
    }

    if(!flag){
        mfem::mfem_warning("Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }

    return tt;
}


}

class CoeffHoles:public mfem::Coefficient
{
public:
    CoeffHoles(double pr=0.5)
    {
        period=pr;
    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {

        double x[3];
        mfem::Vector transip(x, T.GetSpaceDim());
        T.Transform(ip, transip);

        int nx=x[0]/period;
        int ny=x[1]/period;

        x[0]=x[0]-double(nx)*period-0.5*period;
        x[1]=x[1]-double(ny)*period-0.5*period;

        double r=sqrt(x[0]*x[0]+x[1]*x[1]);
        if(r<(0.45*period)){return 0.2;}
        return 0.8;
    }


private:
    double period;
};

class AlcoaBracket
{
public:
    AlcoaBracket(mfem::ParMesh* pmesh, int vorder=1,int seed=std::numeric_limits<int>::max()):E(),nu(0.2)
    {
        esolv=new mfem::ElasticitySolver(pmesh,vorder);
        esolv->AddMaterial(new mfem::LinIsoElasticityCoefficient(E,nu));
        esolv->SetNewtonSolver(1e-8,1e-12,1,0);
        esolv->SetLinearSolver(1e-10,1e-12,400);

        dfes=nullptr;
        cobj=new mfem::ComplianceObjective();

        generator.seed(seed);

        ppmesh=pmesh;

    }

    void SetDesignFES(mfem::ParFiniteElementSpace* fes)
    {
        dfes=fes;
        pdens.SetSpace(dfes);
        vdens.SetSize(dfes->GetTrueVSize());
    }

    ~AlcoaBracket()
    {
        delete cobj;
        delete esolv;

    }

    double Compliance(std::bitset<20>& supp, double eta, mfem::Vector& grad)
    {
        E.SetProjParam(eta,8.0);
        //set all bc
        esolv->DelDispBC();
        for(int j=0;j<20;j++){
            if(supp[j]==true){esolv->AddDispBC(3+j,4,0.0);}
        }
        esolv->AddSurfLoad(1,0.00,1.00,0.0);
        esolv->FSolve();
        esolv->GetSol(sol);

        cobj->Grad(sol,grad);
        return cobj->Eval(sol);
    }

    double Compliance(mfem::Vector& grad, double eta=0.5)
    {
        std::bitset<20> supp;
        for(int i=0;i<20;i++){supp[i]=true;}
        return Compliance(supp,eta,grad);
    }


    double MeanCompl(mfem::Vector& grad)
    {
        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;

        double rez=0.0;
        for(size_t i=0;i<primp.size();i++){
            rez=rez+primp[i]*Compliance(vsupp[i],thresholds[i],cgrad);
            grad.Add(primp[i],cgrad);
        }
        return rez;
    }


    void SetDensity(mfem::Vector& vdens_,
                    double eta=0.5, double beta=8.0,double pen=3.0){

        vdens=vdens_;
        pdens.SetFromTrueDofs(vdens);

        E.SetDens(&pdens);
        E.SetProjParam(eta,beta);
        E.SetEMaxMin(1e-6,1.0);
        E.SetPenal(pen);

        cobj->SetE(&E);
        cobj->SetDens(vdens);
        cobj->SetDesignFES(dfes);

    }

    std::vector<double>& GetDualProb(){	return dualq;}
    std::vector<double>& GetThesholds(){ return thresholds;}
    std::vector<std::bitset<20>>& GetSupp(){ return vsupp;}


    //clears all simulation cases and the associated probabilities
    void ClearCases()
    {
        vsupp.clear();
        dualq.clear();
        thresholds.clear();
    }

    void SetCases2(double eta=0.5)
    {
        std::bitset<20> bset;
        std::bitset<20> aset;

        for(int i=0;i<20;i++){
            for(int j=(i+1);j<20;j++){
                        for(int p=0;p<20;p++){bset[p]=true; aset[p]=true;}
                        bset[i]=false;
                        aset[19-i]=false;
                        bset[j]=false;
                        aset[19-j]=false;
                        vsupp.push_back(bset);
                        asupp.push_back(aset);
                        thresholds.push_back(eta);
                        dualq.push_back(0.1);
            }
        }

        primp.resize(dualq.size());
        //normalize
        double sum=0.0;
        for(size_t i=0;i<dualq.size();i++){sum=sum+dualq[i];}
        for(size_t i=0;i<dualq.size();i++){
            dualq[i]=dualq[i]/sum;
            primp[i]=dualq[i];
        }

        aind.resize(asupp.size());
        for(size_t i=0;i<asupp.size();i++){
            for(size_t j=0;j<vsupp.size();j++){
                if(asupp[i]==vsupp[j]){
                    aind[i]=j;
                    break;
                }
            }
        }

    }


    void SetCases3(double eta=0.5)
    {
        std::bitset<20> bset;
        std::bitset<20> aset;

        for(int i=0;i<20;i++){
            for(int j=(i+1);j<20;j++){
                for(int k=(j+1);k<20;k++){
                        for(int p=0;p<20;p++){bset[p]=true; aset[p]=true;}
                        bset[i]=false;
                        aset[19-i]=false;
                        bset[j]=false;
                        aset[19-j]=false;
                        bset[k]=false;
                        aset[19-k]=false;
                        vsupp.push_back(bset);
                        asupp.push_back(aset);
                        thresholds.push_back(eta);
                        dualq.push_back(0.1);
                }
            }
        }

        primp.resize(dualq.size());
        //normalize
        double sum=0.0;
        for(size_t i=0;i<dualq.size();i++){sum=sum+dualq[i];}
        for(size_t i=0;i<dualq.size();i++){
            dualq[i]=dualq[i]/sum;
            primp[i]=dualq[i];
        }

        aind.resize(asupp.size());
        for(size_t i=0;i<asupp.size();i++){
            for(size_t j=0;j<vsupp.size();j++){
                if(asupp[i]==vsupp[j]){
                    aind[i]=j;
                    break;
                }
            }
        }

    }


    //sets the simulation cases and the associated probabilities
    void SetCases(double eta=0.5)
    {
       std::bitset<20> bset;
       std::bitset<20> aset;

       /*
       for(int j=0;j<20;j++){bset[j]=true;}
       vsupp.push_back(bset);
       asupp.push_back(bset);
       thresholds.push_back(eta);
       dualq.push_back(0.1);

       for(int i=0;i<20;i++){
           for(int j=0;j<20;j++){

               if(j!=i){bset[j]=true;}
               else{bset[j]=false;}

               aset[19-j]=bset[j];
           }

           vsupp.push_back(bset);
           asupp.push_back(aset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }

       for(int i=0;i<19;i++){
           for(int j=0;j<20;j++){
               if((j!=i)&&(j!=(i+1))){bset[j]=true;}
               else{bset[j]=false;}

               aset[19-j]=bset[j];
           }
           vsupp.push_back(bset);
           asupp.push_back(aset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }

       for(int i=0;i<18;i++){
           for(int j=0;j<20;j++){
               if((j!=i)&&(j!=(i+1))&&(j!=(i+2))){bset[j]=true;}
               else{bset[j]=false;}

               aset[19-j]=bset[j];
           }
           vsupp.push_back(bset);
           asupp.push_back(aset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }

       for(int i=0;i<17;i++){
           for(int j=0;j<20;j++){
               if((j!=i)&&(j!=(i+1))&&(j!=(i+2))){
                   if(j!=(i+3)){
                       bset[j]=true;
                   }else{
                       bset[j]=false;
                   }
               }
               else{
                   bset[j]=false;
               }
               aset[19-j]=bset[j];
           }
           vsupp.push_back(bset);
           asupp.push_back(aset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }

       for(int i=0;i<16;i++){
           for(int j=0;j<20;j++){
               if((j!=i)&&(j!=(i+1))&&(j!=(i+2))){
                   if((j!=(i+3))&&(j!=(i+4))){
                       bset[j]=true;
                   }else{
                       bset[j]=false;
                   }
               }
               else{
                   bset[j]=false;
               }
               aset[19-j]=bset[j];
           }
           vsupp.push_back(bset);
           asupp.push_back(aset);
           thresholds.push_back(eta);
           dualq.push_back(0.1);
       }
       */

       for(int i=0;i<20;i++){
           for(int j=(i+1);j<20;j++){
               for(int k=(j+1);k<20;k++){
                   for(int l=(k+1);l<20;l++){
                       for(int p=0;p<20;p++){bset[p]=true; aset[p]=true;}
                       bset[i]=false;
                       aset[19-i]=false;
                       bset[j]=false;
                       aset[19-j]=false;
                       bset[k]=false;
                       aset[19-k]=false;
                       bset[l]=false;
                       aset[19-l]=false;
                       vsupp.push_back(bset);
                       asupp.push_back(aset);
                       thresholds.push_back(eta);
                       dualq.push_back(0.1);
                   }
               }
           }
       }


       primp.resize(dualq.size());
       //normalize
       double sum=0.0;
       for(size_t i=0;i<dualq.size();i++){sum=sum+dualq[i];}
       for(size_t i=0;i<dualq.size();i++){
           dualq[i]=dualq[i]/sum;
           primp[i]=dualq[i];
       }

       aind.resize(asupp.size());
       for(size_t i=0;i<asupp.size();i++){
           for(size_t j=0;j<vsupp.size();j++){
               if(asupp[i]==vsupp[j]){
                   aind[i]=j;
                   break;
               }
           }
       }

       /*
       int myrank=ppmesh->GetMyRank();
       if(myrank==0){
           for(int i=0;i<vsupp.size();i++){
               std::cout<<vsupp[i]<<" "<<asupp[i]<<" a="<<aind[i]<<std::endl;
           }
       }
       */
    }

    double EvalApproxGradientFullSampling(mfem::Vector& grad, double alpha, double gamma)
    {
        int myrank=ppmesh->GetMyRank();
        //compute the objective and the gradients
        int nsampl=dualq.size();
        std::vector<double> vals; vals.resize(nsampl);

        grad=0.0;
        mfem::Vector cgrad(grad.Size()); cgrad=0.0;
        std::vector<mfem::Vector> grads;
        std::vector<int> ind;
        double rez=0.0;
        double nfa=0.0;
        for(size_t i=0;i<dualq.size();i++){
            vals[i]=Compliance(vsupp[i],thresholds[i],cgrad);
            rez=rez+vals[i]*dualq[i];
            grads.push_back(cgrad); ind.push_back(i);
            nfa=nfa+dualq[i];
        }
        rez/=nfa;

        if(myrank==0){
            std::cout<<"rez="<<rez<<" nsampl="<<nsampl<<std::endl;
        }

        //generate qnew
        //find t
        double t;
        if(myrank==0){
            //t=adsampl::Find_t(primp,dualq,alpha,gamma,vals,ind, 1e-12,100);
            t=adsampl::Find_t_bisection(primp,dualq,alpha,gamma,vals,ind, 1e-12,1000);
        }
        //communicate t from 0 to all
        MPI_Bcast(&t,1,MPI_DOUBLE,0,ppmesh->GetComm());

        std::vector<double> w; w.resize(ind.size());
        double tmp;
        for(size_t i=0;i<ind.size();i++){
            tmp=adsampl::inv_sigmoid(dualq[ind[i]],primp[ind[i]],alpha);
            tmp=tmp+gamma*vals[i]-t;
            w[i]=adsampl::sigmoid(tmp,primp[ind[i]],alpha);
        }


        rez=0.0;
        double sum=0.0;
        for(size_t i=0;i<ind.size();i++){
            grad.Add(w[i],grads[i]);
            rez=rez+w[i]*vals[i];
            sum=sum+w[i];
            //copy w to q
            dualq[i]=w[i];
        }
        grad*=gamma;

        if(myrank==0){
            std::cout<<"rez="<<rez<<" t="<<t<<" sum="<<sum<<std::endl;
        }

        //return the objective
        return rez;
    }


    double EvalApproxGradientSampling(mfem::Vector& grad, double alpha, double gamma,int dnsampl)
    {

       int myrank=ppmesh->GetMyRank();
       MPI_Comm comm=ppmesh->GetComm();


       std::vector<int> ind;
       std::vector<int> frq;//frequency of the indices
       int nsampl;
       if(myrank==0){
           std::map<int,int> ind_sampl;
           //do the sampling
           //construct discrete distribution
           std::discrete_distribution<int> d(dualq.begin(),dualq.end());
           for(int i=0;i<dnsampl;i++){
               int vv=d(generator);
               auto it=ind_sampl.find(vv);
               if(it==ind_sampl.end()){
                   ind_sampl[vv]=1;
               }else{
                   it->second=it->second+1;
               }

               it=ind_sampl.find(aind[vv]);
               if(it==ind_sampl.end()){
                   ind_sampl[aind[vv]]=1;
               }else{
                   it->second=it->second+1;
               }

           }

           for(auto it=ind_sampl.begin();it!=ind_sampl.end();it++){
               ind.push_back(it->first);
               frq.push_back(it->second);
           }

           nsampl=ind.size();
       }

       //communicate the samples
       MPI_Bcast(&nsampl,1,MPI_INT,0,comm);
       if(myrank!=0){
           ind.resize(nsampl);
           frq.resize(nsampl);
       }
       MPI_Bcast(ind.data(), nsampl, MPI_INT, 0, comm);
       MPI_Bcast(frq.data(), nsampl, MPI_INT, 0, comm);


       //compute the objective and the gradients
       std::vector<double> vals; vals.resize(nsampl);
       grad=0.0;
       mfem::Vector cgrad(grad.Size()); cgrad=0.0;
       std::vector<mfem::Vector> grads;

       double rez=0.0;
       double nfa=0.0;
       for(int i=0;i<nsampl;i++){
           //generate sample
           vals[i]=Compliance(vsupp[ind[i]],thresholds[ind[i]],cgrad);
           rez=rez+vals[i]*frq[i];
           nfa=nfa+frq[i];
           grads.push_back(cgrad);
       }
       rez=rez/nfa;
       if(myrank==0){
           std::cout<<"rez="<<rez<<" nsampl="<<nsampl<<std::endl;
       }

       //generate qnew
       //find t
       double t;
       if(myrank==0){
           //t=adsampl::Find_t(primp,dualq,alpha,gamma,vals,ind, 1e-12,100);
           t=adsampl::Find_t_bisection(primp,dualq,alpha,gamma,vals,ind, 1e-12,1000);
       }
       //communicate t from 0 to all
       MPI_Bcast(&t,1,MPI_DOUBLE,0,ppmesh->GetComm());

       std::vector<double> w; w.resize(nsampl);
       double tmp;
       for(size_t i=0;i<ind.size();i++){
           tmp=adsampl::inv_sigmoid(dualq[ind[i]],primp[ind[i]],alpha);
           tmp=tmp+gamma*vals[i]-t;
           w[i]=adsampl::sigmoid(tmp,primp[ind[i]],alpha);
       }

       if(myrank==0){
           std::cout<<"w= ";
           for(size_t i=0;i<frq.size();i++){
               std::cout<<" "<<w[i];
           }
           std::cout<<std::endl;
           std::cout<<"frq= ";
           for(size_t i=0;i<frq.size();i++){
               std::cout<<" "<<frq[i];
           }
           std::cout<<std::endl;
       }

       rez=0.0;
       for(size_t i=0;i<ind.size();i++){
           double lw=frq[i]*w[i]/dualq[ind[i]];
           grad.Add(lw,grads[i]);
           rez=rez+lw*vals[i];
           //copy w to q
           dualq[ind[i]]=w[i];
       }


       rez=rez/nfa;
       grad*=(gamma/nfa);


       if(myrank==0){
           std::cout<<"rez="<<rez<<" t="<<t<<std::endl;
       }

       return rez;
    }

    double EvalApproxGradientSamplingMem(mfem::Vector& grad, double alpha, double gamma,int dnsampl)
    {

       int myrank=ppmesh->GetMyRank();
       MPI_Comm comm=ppmesh->GetComm();


       std::vector<int> ind;
       std::vector<int> frq;//frequency of the indices
       int nsampl;
       if(myrank==0){
           std::map<int,int> ind_sampl;
           //do the sampling
           //construct discrete distribution
           std::discrete_distribution<int> d(dualq.begin(),dualq.end());
           for(int i=0;i<dnsampl;i++){
               int vv=d(generator);
               auto it=ind_sampl.find(vv);
               if(it==ind_sampl.end()){
                   ind_sampl[vv]=1;
               }else{
                   it->second=it->second+1;
               }

               it=ind_sampl.find(aind[vv]);
               if(it==ind_sampl.end()){
                   ind_sampl[aind[vv]]=1;
               }else{
                   it->second=it->second+1;
               }

           }

           for(auto it=ind_sampl.begin();it!=ind_sampl.end();it++){
               ind.push_back(it->first);
               frq.push_back(it->second);
           }

           nsampl=ind.size();
       }

       //communicate the samples
       MPI_Bcast(&nsampl,1,MPI_INT,0,comm);
       if(myrank!=0){
           ind.resize(nsampl);
           frq.resize(nsampl);
       }
       MPI_Bcast(ind.data(), nsampl, MPI_INT, 0, comm);
       MPI_Bcast(frq.data(), nsampl, MPI_INT, 0, comm);


       //compute the objective and the gradients
       std::vector<double> vals; vals.resize(nsampl);
       grad=0.0;
       mfem::Vector cgrad(grad.Size()); cgrad=0.0;
       //std::vector<mfem::Vector> grads;

       double rez=0.0;
       double nfa=0.0;
       for(int i=0;i<nsampl;i++){
           //generate sample
           vals[i]=Compliance(vsupp[ind[i]],thresholds[ind[i]],cgrad);
           rez=rez+vals[i]*frq[i];
           nfa=nfa+frq[i];
           //grads.push_back(cgrad);
       }
       rez=rez/nfa;
       if(myrank==0){
           std::cout<<"rez="<<rez<<" nsampl="<<nsampl<<" q_size="<<dualq.size()<<std::endl;
       }

       //generate qnew
       //find t
       double t;
       if(myrank==0){
           //t=adsampl::Find_t(primp,dualq,alpha,gamma,vals,ind, 1e-12,100);
           t=adsampl::Find_t_bisection(primp,dualq,alpha,gamma,vals,ind, 1e-12,1000);
       }
       //communicate t from 0 to all
       MPI_Bcast(&t,1,MPI_DOUBLE,0,ppmesh->GetComm());

       std::vector<double> w; w.resize(nsampl);
       double tmp;
       for(size_t i=0;i<ind.size();i++){
           tmp=adsampl::inv_sigmoid(dualq[ind[i]],primp[ind[i]],alpha);
           tmp=tmp+gamma*vals[i]-t;
           w[i]=adsampl::sigmoid(tmp,primp[ind[i]],alpha);
       }

       /*
       if(myrank==0){
           std::cout<<"w= ";
           for(size_t i=0;i<frq.size();i++){
               std::cout<<" "<<w[i];
           }
           std::cout<<std::endl;
           std::cout<<"frq= ";
           for(size_t i=0;i<frq.size();i++){
               std::cout<<" "<<frq[i];
           }
           std::cout<<std::endl;
       }
       */

       rez=0.0;
       for(size_t i=0;i<ind.size();i++){
           double lw=frq[i]*w[i]/dualq[ind[i]];
           //recompute cgrad
           vals[i]=Compliance(vsupp[ind[i]],thresholds[ind[i]],cgrad);
           grad.Add(lw,cgrad);
           rez=rez+lw*vals[i];
           //copy w to q
           dualq[ind[i]]=w[i];
       }


       rez=rez/nfa;
       grad*=(gamma/nfa);


       if(myrank==0){
           std::cout<<"rez="<<rez<<" t="<<t<<std::endl;
       }

       return rez;
    }

private:
    mfem::YoungModulus E;
    double nu;

    mfem::ParFiniteElementSpace* dfes; //design FES
    mfem::ParGridFunction pdens;
    mfem::Vector vdens;

    mfem::ElasticitySolver* esolv;
    mfem::ComplianceObjective* cobj;

    mfem::ParGridFunction sol;


    //the following three vectors should have the same size
    std::vector<double> dualq;
    std::vector<double> primp;
    std::vector<std::bitset<20>> vsupp;
    std::vector<std::bitset<20>> asupp;
    std::vector<int> aind;
    std::vector<double> thresholds;

    std::default_random_engine generator;
    mfem::ParMesh* ppmesh;
};



int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "./canti_2D_m.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 51;
   int print_level = 1;
   bool visualization = false;
   const char *petscrc_file = "";
   int restart=0;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&tot_iter,
                  "-it",
                  "--linear-iterations",
                  "Maximum iterations for the linear solve.");
   args.AddOption(&max_it,
                  "-mit",
                  "--max-optimization-iterations",
                  "Maximum iterations for the linear optimizer.");
   args.AddOption(&fradius,
                  "-r",
                  "--radius",
                  "Filter radius");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
   args.AddOption(&restart,
                     "-rstr",
                     "--restart",
                     "Restart the optimization from previous design.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }
   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   {
       mfem::Vector vert;
       mesh.GetVertices(vert);
       vert*=0.01;
       mesh.SetVertices(vert);
       mfem::Vector xmin(dim), xmax(dim);
       mesh.GetBoundingBox(xmin,xmax);
       if(myrank==0){
           std::cout<<"Xmin:";xmin.Print(std::cout);
           std::cout<<"Xmax:";xmax.Print(std::cout);
       }
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh.UniformRefinement();
       }
   }

   if(myrank==0)
   {
       std::cout<<"num el="<<pmesh.GetNE()<<std::endl;
   }


   //allocate the filter
   mfem::FilterSolver* fsolv=new mfem::FilterSolver(0.07,&pmesh);
   fsolv->SetSolver(1e-8,1e-12,100,0);
   fsolv->AddBC(1,1.0);
   fsolv->AddBC(2,0.0);
   for(int i=3;i<23;i++){
       fsolv->AddBC(i,1.0);
   }

   mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
   mfem::ParGridFunction oddens(fsolv->GetDesignFES());
   mfem::ParGridFunction spdegf(fsolv->GetFilterFES());
   mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
   mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

   fsolv->Mult(vtmpv,vdens);
   pgdens.SetFromTrueDofs(vdens);


   AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
   alco->SetDesignFES(pgdens.ParFESpace());
   alco->SetDensity(vdens);
   alco->SetCases2(0.7);

   //mfem::ParGridFunction disp;
   //alco->GetSol(4,1,1,1,disp);

   //check gradients
   /*
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;
       mfem::Vector tgrad;
       mfem::Vector fgrad;
       prtv.SetSize(vtmpv.Size());
       tmpv.SetSize(vtmpv.Size());
       tgrad.SetSize(vtmpv.Size());
       fgrad.SetSize(vdens.Size()); fgrad=0.0;
       double val=alco->MeanCompliance();
       alco->MeanCompliance(fgrad);
       fsolv->MultTranspose(fgrad,tgrad);

       prtv.Randomize();
       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);
       td=td/nd;
       double lsc=1.0;
       double lqoi;

       for(int l=0;l<10;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,vtmpv,tmpv);
           fsolv->Mult(tmpv,vdens);
           alco->SetDensity(vdens);
           alco->Solve();
           lqoi=alco->MeanCompliance();
           double ld=(lqoi-val)/lsc;
           if(myrank==0){
               std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                         << " adjoint gradient=" << td
                         << " err=" << std::fabs(ld/nd-td) << std::endl;
           }
       }
   }*/

   mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
   //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   vobj->SetProjection(0.5,8.0);//threshold 0.2

   //compute the total volume
   double tot_vol;
   {
       vdens=1.0;
       tot_vol=vobj->Eval(vdens);
   }
   double max_vol=0.25*tot_vol;
   if(myrank==0){ std::cout<<"tot vol="<<tot_vol<<std::endl;}

   //intermediate volume
   mfem::VolumeQoI* ivobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
   ivobj->SetProjection(0.5,32);

   //gradients with respect to the filtered field
   mfem::Vector ograd(fsolv->GetFilterFES()->GetTrueVSize()); ograd=0.0; //of the objective
   mfem::Vector vgrad(fsolv->GetFilterFES()->GetTrueVSize()); vgrad=0.0; //of the volume contr.

   //the input design field and the filtered one might not have the same dimensionality
   mfem::Vector ogrado(fsolv->GetDesignFES()->GetTrueVSize()); ogrado=0.0;
   mfem::Vector vgrado(fsolv->GetDesignFES()->GetTrueVSize()); vgrado=0.0;

   mfem::Vector xxmax(fsolv->GetDesignFES()->GetTrueVSize()); xxmax=1.0;
   mfem::Vector xxmin(fsolv->GetDesignFES()->GetTrueVSize()); xxmin=0.0;

   mfem::NativeMMA* mma;
   {
       double a=0.0;
       double c=1000.0;
       double d=0.0;
       mma=new mfem::NativeMMA(MPI_COMM_WORLD,1, ogrado,&a,&c,&d);
   }

   double max_ch=0.1; //max design change

   double cpl; //compliance
   double vol; //volume
   double ivol; //intermediate volume


   mfem::ParGridFunction solx;
   mfem::ParGridFunction soly;

   {
      mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);

      paraview_dc.RegisterField("design",&pgdens);

      //alco->GetSol(6,0.0,1.0,0.0,solx);
      //alco->GetSol(1,0.0,1.0,0.0,soly);
      //paraview_dc.RegisterField("solx",&solx);
      //paraview_dc.RegisterField("soly",&soly);

      //spdegf.ProjectCoefficient(spderf);
      //paraview_dc.RegisterField("reta",&spdegf);

      paraview_dc.Save();

      CoeffHoles holes;
      oddens.ProjectCoefficient(holes);
      oddens.GetTrueDofs(vtmpv);
      vtmpv=0.3;
      fsolv->Mult(vtmpv,vdens);
      pgdens.SetFromTrueDofs(vdens);


      for(int i=1;i<max_it;i++){

          vobj->SetProjection(0.3,8.0);
          alco->SetDensity(vdens,0.7,8.0,1.0);

          //cpl=alco->Compliance(ograd);
          //cpl=alco->MeanCompl(ograd);
          //cpl=alco->EGDUpdate(ograd,0.001);

          //cpl=alco->EvalApproxGradientFullSampling(ograd,0.90,0.01);
          //cpl=alco->EvalApproxGradientSampling(ograd,0.90,0.001,20);
          cpl=alco->EvalApproxGradientSamplingMem(ograd,0.90,0.1,190*4);
          vol=vobj->Eval(vdens);
          ivol=ivobj->Eval(vdens);


          if(myrank==0){
              std::cout<<"it: "<<i<<" obj="<<cpl<<" vol="<<vol<<" cvol="<<max_vol<<" ivol="<<ivol
                      <<std::endl;
          }
          vobj->Grad(vdens,vgrad);
          //compute the original gradients
          fsolv->MultTranspose(ograd,ogrado);
          fsolv->MultTranspose(vgrad,vgrado);

          {
              //set xxmin and xxmax
              xxmin=vtmpv; xxmin-=max_ch;
              xxmax=vtmpv; xxmax+=max_ch;
              for(int li=0;li<xxmin.Size();li++){
                  if(xxmin[li]<0.0){xxmin[li]=0.0;}
                  if(xxmax[li]>1.0){xxmax[li]=1.0;}
              }
          }

          double con=vol-max_vol;
          mma->Update(vtmpv,ogrado,&con,&vgrado,xxmin,xxmax);

          fsolv->Mult(vtmpv,vdens);
          pgdens.SetFromTrueDofs(vdens);

          //alco->GetSol(1,0.0,1.0,0.0,solx);
          //alco->GetSol(6,0.0,1.0,0.0,soly);

          //paraview_dc.RegisterField("solx",&solx);
          //paraview_dc.RegisterField("soly",&soly);

          //save the design
          if(i%4==0)
          {
              paraview_dc.SetCycle(i);
              paraview_dc.SetTime(i*1.0);
              paraview_dc.Save();
          }
      }

   }


   delete mma;
   delete vobj;
   delete ivobj;
   delete alco;
   delete fsolv;

   mfem::MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}
