#ifndef STOKES_HPP
#define STOKES_HPP

#include "mfem.hpp"
#include "admfem.hpp"


namespace mfem {

namespace PointwiseTrans
{

/*  Standrd "Heaviside" projection in topology optimization with threhold eta
 * and steepness of the projection beta.
 * */
inline
double HProject(double rho, double eta, double beta)
{
    // tanh projection - Wang&Lazarov&Sigmund2011
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double c=std::tanh(beta*(rho-eta));
    double rez=(a+c)/(a+b);
    return rez;
}

/// Gradient of the "Heaviside" projection with respect to rho.
inline
double HGrad(double rho, double eta, double beta)

{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=beta*(1.0-c*c)/(a+b);
    return rez;
}

/// Second derivative of the "Heaviside" projection with respect to rho.
inline
double HHess(double rho,double eta, double beta)
{
    double c=std::tanh(beta*(rho-eta));
    double a=std::tanh(eta*beta);
    double b=std::tanh(beta*(1.0-eta));
    double rez=-2.0*beta*beta*c*(1.0-c*c)/(a+b);
    return rez;
}


inline
double FluidInterpolation(double rho,double q)
{
    return q*(1.0-rho)/(q+rho);
}

inline
double GradFluidInterpolation(double rho, double q)
{
    double tt=q+rho;
    return -q/tt-q*(1.0-rho)/(tt*tt);
}


}




// Taylor-Hood finite elements
class StokesIntegratorTH:public mfem::BlockNonlinearFormIntegrator
{
public:
    StokesIntegratorTH()
    {
        mu=nullptr;
        bc=nullptr;
        ff=nullptr;

        ss.SetSize(13);
        rr.SetSize(13);
        mm.SetSize(13);
    }

    StokesIntegratorTH(mfem::Coefficient* mu_, mfem::Coefficient* bc_, mfem::VectorCoefficient* ff_)
    {
        mu=mu_;
        bc=bc_;
        ff=ff_;

        ss.SetSize(13);
        rr.SetSize(13);
        mm.SetSize(13);
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
                               const Array<Vector *> &elvec)
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();

        int dim =  el[0]->GetDim();

        elvec[0]->SetSize(dim*dof_u);
        elvec[1]->SetSize(dof_p);

        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("StokesIntegrator::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        // gradients
        bsu.SetSize(dof_u,4);
        bsp.SetSize(dof_p,1);

        Vector uu(elfun[0]->GetData()+0*dof_u, dof_u);
        Vector vv(elfun[0]->GetData()+1*dof_u, dof_u);

        Vector ru(elvec[0]->GetData()+0*dof_u, dof_u); ru=0.0;
        Vector rv(elvec[0]->GetData()+1*dof_u, dof_u); rv=0.0;

        Vector ww;
        Vector rw;
        if(dim==2){
            ww.SetSize(dof_u); ww=0.0;
        }
        else{
            ww.SetDataAndSize(elfun[0]->GetData()+2*dof_u, dof_u);
            rw.SetDataAndSize(elvec[0]->GetData()+2*dof_u, dof_u); rw=0.0;
        }

        Vector pp(elfun[1]->GetData(), dof_p);
        Vector rp(elvec[1]->GetData(), dof_p); rp=0.0;

        // temp storages for vectors and matrices
        Vector sh;
        DenseMatrix dh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);

        double bpenal; // Brinkmann penalization
        double mmu;
        Vector fv(3); fv=0.0;
        double w;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();
           w = ip.weight * w;

           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);
           sh.SetDataAndSize(bsp.GetData(),dof_p);
           el[1]->CalcPhysShape(Tr,sh);

           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);
           if(dim=2){
               sh.SetDataAndSize(bsu.GetData()+3*dof_u,dof_u);
               sh=0.0;
           }

           sh.SetDataAndSize(ss.GetData(),4);
           bsu.MultTranspose(uu,sh);
           sh.SetDataAndSize(ss.GetData()+4,4);
           bsu.MultTranspose(vv,sh);
           sh.SetDataAndSize(ss.GetData()+8,4);
           if(dim==3){
               bsu.MultTranspose(ww,sh);}
           else{
               sh=0.0;}
           sh.SetDataAndSize(ss.GetData()+12,1);
           bsp.MultTranspose(pp,sh);

           mmu=1.0;
           if(mu!=nullptr){
               mmu=mu->Eval(Tr,ip);}

           bpenal=0.0;
           if(bc!=nullptr){
               bpenal=bc->Eval(Tr,ip);}

           fv=0.0;
           if(ff!=nullptr){
               ff->Eval(fv,Tr,ip);}

           EvalQres(mmu,bpenal,fv[0],fv[1],fv[2],ss.GetData(),rr.GetData());

           sh.SetDataAndSize(rr.GetData(),4);
           bsu.AddMult_a(w,sh,ru);
           sh.SetDataAndSize(rr.GetData()+4,4);
           bsu.AddMult_a(w,sh,rv);
           if(dim==3){
               sh.SetDataAndSize(rr.GetData()+8,4);
               bsu.AddMult_a(w,sh,rw);
           }
           sh.SetDataAndSize(rr.GetData()+12,1);
           bsp.AddMult_a(w,sh,rp);
        }

    }

    virtual
    void AssembleElementGrad(const Array<const FiniteElement *> &el,
                             ElementTransformation &Tr,
                             const Array<const Vector *> &elfun,
                             const Array2D<DenseMatrix *> &elmats)
    {
        int dof_u = el[0]->GetDof();
        int dof_p = el[1]->GetDof();

        int dim =  el[0]->GetDim();

        int spaceDim = Tr.GetDimension();
        if (dim != spaceDim)
        {
           mfem::mfem_error("StokesIntegrator::AssembleElementVector"
                            " is not defined on manifold meshes");
        }

        elmats(0,0)->SetSize(dof_u*dim);
        elmats(0,1)->SetSize(dof_u*dim,dof_p);
        elmats(1,0)->SetSize(dof_p,dim*dof_u);
        elmats(1,1)->SetSize(dof_p,dof_p);

        (*elmats(0,0))=0.0;
        (*elmats(0,1))=0.0;
        (*elmats(1,0))=0.0;
        (*elmats(1,1))=0.0;

        // gradients
        DenseMatrix bsu(dof_u,4);
        DenseMatrix bsp(dof_p,1);

        mm.SetSize(13,13); // state matrix

        // temp storages for vectors and matrices
        Vector sh;
        DenseMatrix mh;
        DenseMatrix th;
        DenseMatrix rh;
        DenseMatrix dh;

        const IntegrationRule *ir = nullptr;
        int order= 2 * el[0]->GetOrder() + Tr.OrderGrad(el[0]);
        ir=&IntRules.Get(Tr.GetGeometryType(),order);


        double mmu;
        double bpenal; // Brinkmann penalization
        double w;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();

           // Primal
           sh.SetDataAndSize(bsu.GetData(),dof_u);
           el[0]->CalcPhysShape(Tr,sh);
           sh.SetDataAndSize(bsp.GetData(),dof_p);
           el[1]->CalcPhysShape(Tr,sh);


           // Gradients
           dh.UseExternalData(bsu.GetData()+dof_u,dof_u,dim);
           el[0]->CalcPhysDShape(Tr,dh);

           if(dim=2){
               sh.SetDataAndSize(bsu.GetData()+3*dof_u,dof_u);
               sh=0.0;
           }

           mmu=1.0;
           if(mu!=nullptr)
           {
               mmu=mu->Eval(Tr,ip);
           }

           bpenal=0.0;
           if(bc!=nullptr){
               bpenal=bc->Eval(Tr,ip);}

           EvalQMat(mmu,bpenal,mm.GetData());

           w = ip.weight * w;

           th.SetSize(dof_u,4);
           rh.SetSize(dof_u);
           mh.SetSize(4,4);
           for(int ii=0;ii<dim;ii++){
           for(int jj=0;jj<dim;jj++){
               mh.CopyMN(mm,4,4,ii*4,jj*4);
               mh.Transpose();
               MultABt(bsu,mh,th);
               MultABt(th,bsu,rh);
               elmats(0,0)->AddMatrix(w,rh,ii*dof_u,jj*dof_u);
           }}

           th.SetSize(dof_u,1);
           rh.SetSize(dof_u,dof_p);
           mh.SetSize(4,1);
           for(int jj=0;jj<dim;jj++){
               mh.CopyMN(mm,4,1,jj*4,12);
               mh.Transpose();
               MultABt(bsu,mh,th);
               MultABt(th,bsp,rh);
               elmats(0,1)->AddMatrix(w,rh,jj*dof_u,0);
           }

           th.SetSize(dof_p,1);
           rh.SetSize(dof_p,dof_p);
           mh.SetSize(1,1);
           mh.CopyMN(mm,1,1,12,12);
           mh.Transpose();
           MultABt(bsp,mh,th);
           MultABt(th,bsp,rh);
           elmats(1,1)->AddMatrix(w,rh,0,0);

        }

        elmats(1,0)->CopyMNt(*elmats(0,1),0,0);
    }


    void SetParameters(mfem::Coefficient* mu_, mfem::Coefficient* bc_, mfem::VectorCoefficient* ff_)
    {
        mu=mu_;
        bc=bc_;
        ff=ff_;
    }

private:
    mfem::Coefficient* mu;
    mfem::Coefficient* bc;
    mfem::VectorCoefficient* ff;

    mfem::DenseMatrix bsu;
    mfem::DenseMatrix bsp;
    DenseMatrix mm;
    Vector rr;
    Vector ss;

    void EvalQres(double mmu, double bpenal,
                  double fx, double fy, double fz,
                  double* uu, double* rr)
    {
        double t7,t9,t16;
        t7 = mmu*(uu[2]+uu[5]);
        t9 = mmu*(uu[3]+uu[9]);
        t16 = mmu*(uu[7]+uu[10]);
        rr[0] = bpenal*uu[0]-fx;
        rr[1] = 2.0*mmu*uu[1]-uu[12];
        rr[2] = t7;
        rr[3] = t9;
        rr[4] = bpenal*uu[4]-fy;
        rr[5] = t7;
        rr[6] = 2.0*mmu*uu[6]-uu[12];
        rr[7] = t16;
        rr[8] = bpenal*uu[8]-fz;
        rr[9] = t9;
        rr[10] = t16;
        rr[11] = 2.0*mmu*uu[11]-uu[12];
        rr[12] = -uu[1]-uu[6]-uu[11];
    }

    void EvalQMat(double mmu, double bpenal, double* kmat)
    {
        for(int i=0;i<169;i++){
            kmat[i]=0.0;
        }
        double t1 = 2.0*mmu;
        kmat[0] = bpenal;
        kmat[14] = t1;
        kmat[25] = -1.0;
        kmat[28] = mmu;
        kmat[31] = mmu;
        kmat[42] = mmu;
        kmat[48] = mmu;
        kmat[56] = bpenal;
        kmat[67] = mmu;
        kmat[70] = mmu;
        kmat[84] = t1;
        kmat[90] = -1.0;
        kmat[98] = mmu;
        kmat[101] = mmu;
        kmat[112] = bpenal;
        kmat[120] = mmu;
        kmat[126] = mmu;
        kmat[137] = mmu;
        kmat[140] = mmu;
        kmat[154] = t1;
        kmat[155] = -1.0;
        kmat[157] = -1.0;
        kmat[162] = -1.0;
        kmat[167] = -1.0;
    }

};

class StokesResidualIntegrator: public LinearFormIntegrator
{
public:
    StokesResidualIntegrator(mfem::Coefficient* mu_, mfem::Coefficient* bc_, mfem::VectorCoefficient* ff_)
    {
        mu=mu_;
        bc=bc_;
        ff=ff_;

        lss.SetSize(13);
        laa.SetSize(13);
        hss.SetSize(13);
        haa.SetSize(13);

        rr.SetSize(13);
    }

    void SetLORFields(mfem::GridFunction& fvel, mfem::GridFunction& fpress,
                      mfem::GridFunction& avel, mfem::GridFunction& apress)
    {
        lorfVelocity=&fvel;
        lorfPressure=&fpress;
        loraVelocity=&avel;
        loraPressure=&apress;
    }

    void SetHORFields(mfem::GridFunction& fvel, mfem::GridFunction& fpress,
                      mfem::GridFunction& avel, mfem::GridFunction& apress)
    {
        horfVelocity=&fvel;
        horfPressure=&fpress;
        horaVelocity=&avel;
        horaPressure=&apress;
    }


    virtual
    void AssembleRHSElementVect(const FiniteElement &el,
                                ElementTransformation &Tr,
                                Vector &elvect)
    {

        const int dof = el.GetDof();
        const int dim = el.GetDim();
        elvect.SetSize(dof); elvect=0.0;
        mfem::Vector sh(dof); // shape functions

        const IntegrationRule *ir = nullptr;
        {
            int io = horfVelocity->FESpace()->GetOrder(Tr.ElementNo);
            int order= 2 * io +  Tr.OrderGrad(&el);
            ir=&IntRules.Get(Tr.GetGeometryType(),order);
        }

        double bpenal=0.0; // Brinkmann penalization
        double mmu=1.0;
        Vector fv(3); fv=0.0;
        double w;

        Vector fvel(dim);
        Vector avel(dim);
        double fpress, apress;

        DenseMatrix vgrad(dim);
        DenseMatrix agrad(dim);


        laa=0.0;
        lss=0.0;
        haa=0.0;
        hss=0.0;

        double res=0.0;
        double dp1,dp2;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           w=Tr.Weight();
           w = ip.weight * w;

           mmu=1.0;
           if(mu!=nullptr){
               mmu=mu->Eval(Tr,ip);}

           bpenal=0.0;
           if(bc!=nullptr){
               bpenal=bc->Eval(Tr,ip);}

           fv=0.0;
           if(ff!=nullptr){
               ff->Eval(fv,Tr,ip);}

           el.CalcPhysShape(Tr,sh);

           //low order solutions
           lorfVelocity->GetVectorValue(Tr,ip,fvel);
           lorfVelocity->GetVectorGradient(Tr,vgrad);

           loraVelocity->GetVectorValue(Tr,ip,avel);
           loraVelocity->GetVectorGradient(Tr,agrad);

           fpress=lorfPressure->GetValue(Tr,ip);
           apress=loraPressure->GetValue(Tr,ip);


           for(int ii=0;ii<dim;ii=ii+1)
           {
               lss[0+4*ii]=fvel[ii];
               laa[0+4*ii]=avel[ii];
           }
           //set the gradients
           for(int ii=0;ii<dim;ii=ii+1){
           for(int jj=0;jj<dim;jj=jj+1){
               lss[jj*4+1+ii]=vgrad(jj,ii);
               laa[jj*4+1+ii]=agrad(jj,ii);
           }}
           lss[12]=fpress;
           laa[12]=apress;


           //high order solutions
           horfVelocity->GetVectorValue(Tr,ip,fvel);
           horfVelocity->GetVectorGradient(Tr,vgrad);

           horaVelocity->GetVectorValue(Tr,ip,avel);
           horaVelocity->GetVectorGradient(Tr,agrad);

           fpress=horfPressure->GetValue(Tr,ip);
           apress=horaPressure->GetValue(Tr,ip);


           for(int ii=0;ii<dim;ii=ii+1)
           {
               hss[0+4*ii]=fvel[ii];
               haa[0+4*ii]=avel[ii];
           }
           //set the gradients
           for(int ii=0;ii<dim;ii=ii+1){
           for(int jj=0;jj<dim;jj=jj+1){
               hss[jj*4+1+ii]=vgrad(jj,ii);
               haa[jj*4+1+ii]=agrad(jj,ii);
           }}
           hss[12]=fpress;
           haa[12]=apress;


           res=0.0;

           EvalQres(mmu,bpenal,fv[0],fv[1],fv[2],lss,rr);
           dp1=-(laa*rr);
           dp2=-(haa*rr);
           res=res+w*(dp2-dp1)*0.5;

           EvalQres(mmu,bpenal,0.0,0.0,0.0,laa,rr);
           dp1=-(lss*rr);
           dp2=-(hss*rr);
           res=res+w*(dp2-dp1)*0.5;

           for(int ii=0;ii<dof;ii++){
                elvect[ii]=elvect[ii]+res*sh[ii];
           }

        }

    }

private:
    mfem::Coefficient* mu;
    mfem::Coefficient* bc;
    mfem::VectorCoefficient* ff;

    mfem::GridFunction* lorfVelocity;
    mfem::GridFunction* loraVelocity;
    mfem::GridFunction* horfVelocity;
    mfem::GridFunction* horaVelocity;
    mfem::GridFunction* lorfPressure;
    mfem::GridFunction* loraPressure;
    mfem::GridFunction* horfPressure;
    mfem::GridFunction* horaPressure;

    Vector rr;
    Vector lss;
    Vector hss;
    Vector laa;
    Vector haa;

    void EvalQres(double mmu, double bpenal,
                  double fx, double fy, double fz,
                  double* uu, double* rr)
    {
        double t7,t9,t16;
        t7 = mmu*(uu[2]+uu[5]);
        t9 = mmu*(uu[3]+uu[9]);
        t16 = mmu*(uu[7]+uu[10]);
        rr[0] = bpenal*uu[0]-fx;
        rr[1] = 2.0*mmu*uu[1]-uu[12];
        rr[2] = t7;
        rr[3] = t9;
        rr[4] = bpenal*uu[4]-fy;
        rr[5] = t7;
        rr[6] = 2.0*mmu*uu[6]-uu[12];
        rr[7] = t16;
        rr[8] = bpenal*uu[8]-fz;
        rr[9] = t9;
        rr[10] = t16;
        rr[11] = 2.0*mmu*uu[11]-uu[12];
        rr[12] = -uu[1]-uu[6]-uu[11];
    }

};


class FluidInterpolationCoefficient: public mfem::Coefficient
{
public:
    FluidInterpolationCoefficient()
    {
        eta=0.5;
        beta=8.0;
        q=1.0;
        lambda=100;
    }

    FluidInterpolationCoefficient(FluidInterpolationCoefficient& coef)
    {
        eta=coef.eta;
        beta=coef.beta;
        q=coef.q;
        lambda=coef.lambda;
        gfc=coef.gfc;
    }

    FluidInterpolationCoefficient(const FluidInterpolationCoefficient& coef)
    {
        eta=coef.eta;
        beta=coef.beta;
        q=coef.q;
        lambda=coef.lambda;
        gfc=coef.gfc;
    }

    FluidInterpolationCoefficient(double eta_, double beta_,
                                  double q_, double lambda_, mfem::GridFunction& gf):gfc(&gf)
    {
        eta=eta_;
        beta=beta_;
        q=q_;
        lambda=lambda_;
    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double rhop=PointwiseTrans::HProject(gfc.Eval(T,ip),eta,beta);
        return lambda*PointwiseTrans::FluidInterpolation(rhop,q);
        //return 1.0-gfc.Eval(T,ip);

    }

    virtual
    double GradEval(ElementTransformation &T, const IntegrationPoint &ip)
    {

        double rhoo=gfc.Eval(T,ip);
        double rhop=PointwiseTrans::HProject(rhoo,eta,beta);
        double g1=lambda*PointwiseTrans::GradFluidInterpolation(rhop,q);
        double g2=PointwiseTrans::HGrad(rhoo,eta,beta);
        return g1*g2;
        //return -1.0;
    }

    void SetGridFunction(mfem::GridFunction* gf)
    {
        gfc.SetGridFunction(gf);
    }

    void SetGridFunction(mfem::GridFunction& gf)
    {
        gfc.SetGridFunction(&gf);
    }

    void SetParameters(double eta_, double beta_,
                       double q_, double lambda_)
    {
        eta=eta_;
        beta=beta_;
        q=q_;
        lambda=lambda_;
    }

    void SetPenalization(double lambda_)
    {
        lambda=lambda_;
    }

    double GetPenalization(){
        return lambda;
    }

    double GetBeta()
    {
        return beta;
    }

    double GetEta()
    {
        return eta;
    }

    void SetBeta(double beta_)
    {
        beta=beta_;
    }

    void SetEta(double eta_)
    {
        eta=eta_;
    }

    double GetQ()
    {
        return q;
    }

    void SetQ(double q_)
    {
        q=q_;
    }


private:
    mfem::GridFunctionCoefficient gfc;
    double eta;  // threshold level
    double beta; // steepness of the projection
    double q; // interpolation parameter
    double lambda; // penalization

};

class StokesModelErrorIntegrator: public mfem::LinearFormIntegrator
{
public:
    StokesModelErrorIntegrator(mfem::GridFunction& velocity,
                               mfem::GridFunction& adjoint,
                               mfem::FluidInterpolationCoefficient& bcoef,
                               mfem::FluidInterpolationCoefficient& tcoef)
                                :vel(velocity), adj(adjoint), brm(bcoef), trm(tcoef)
    {
    }

    virtual
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, mfem::Vector &elvect)
    {
         int dof = el.GetDof();
         int dim = el.GetDim();
         elvect.SetSize(dof);
         elvect=0.0;

         int intorder = vel.FESpace()->GetOrder(Tr.ElementNo);
         const mfem::IntegrationRule *ir = nullptr;
         int order= 3 * intorder + Tr.OrderGrad(&el);
         ir=&IntRules.Get(Tr.GetGeometryType(),order);

         //shape funtions
         mfem::Vector sh(dof);

         //velocity
         mfem::Vector vv(dim);
         //adjoint
         mfem::Vector aa(dim);

         double gbpenal; //difference of the Brinkmann penalizations
         double dp;
         double w;


         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w=Tr.Weight();
            w = ip.weight * w;

            gbpenal=trm.Eval(Tr,ip)-brm.Eval(Tr,ip);

            vel.GetVectorValue(Tr,ip,vv);
            adj.GetVectorValue(Tr,ip,aa);

            el.CalcPhysShape(Tr,sh);
            dp=vv*aa;

            for(int i=0;i<dof;i++)
            {
                elvect[i]=elvect[i]-w*sh[i]*gbpenal*dp;
            }

         }
    }


private:
    mfem::GridFunction &vel;
    mfem::GridFunction &adj;
    mfem::FluidInterpolationCoefficient& brm; //dnesity based interpolation
    mfem::FluidInterpolationCoefficient& trm; //true distribution
};

class StokesGradIntergrator: public mfem::LinearFormIntegrator
{
public:
    StokesGradIntergrator(mfem::GridFunction& velocity,
                          mfem::GridFunction& adjoint,
                          mfem::FluidInterpolationCoefficient& bcoef,
                          int intorder_)
        :vel(velocity), adj(adjoint), brm(bcoef)
    {
        intorder=intorder_; // should be equal to the order of the velocity field
    }

    virtual
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, mfem::Vector &elvect)
    {
         int dof = el.GetDof();
         int dim = el.GetDim();
         elvect.SetSize(dof);
         elvect=0.0;

         const mfem::IntegrationRule *ir = nullptr;
         int order= 2 * intorder + Tr.OrderGrad(&el);
         ir=&IntRules.Get(Tr.GetGeometryType(),order);

         mfem::Vector sh(dof);
         mfem::Vector vv(dim);
         mfem::Vector aa(dim);

         double gbpenal; //gradient of the Brinkmann penalization
         double dp;
         double w;


         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w=Tr.Weight();
            w = ip.weight * w;

            gbpenal=brm.GradEval(Tr,ip);

            vel.GetVectorValue(Tr,ip,vv);
            adj.GetVectorValue(Tr,ip,aa);

            el.CalcPhysShape(Tr,sh);

            for(int i=0;i<dof;i++)
            {
                dp=vv*aa;
                elvect[i]=elvect[i]-w*sh[i]*gbpenal*dp;
            }

         }
    }

private:
    mfem::GridFunction& vel;
    mfem::GridFunction& adj;
    mfem::FluidInterpolationCoefficient& brm;
    int intorder;

};


template<typename TDataType, typename TParamVector, typename TStateVector
         , int state_size, int param_size>
class PowerDissipationFunctional
{
public:
    TDataType operator () (TParamVector& vparam, TStateVector& uu)
    {
        MFEM_ASSERT(state_size==9,"LinElastFunctional state_size should be equal to 9!");
        MFEM_ASSERT(param_size==1,"LinElastFunctional param_size should be equal to 2!");
        auto nu = vparam[0]; //fluid viscosity

        TDataType rez;
        TDataType st[6];
        //strain
        st[0]=uu[0];
        st[1]=uu[4];
        st[2]=uu[8];
        st[3]=(uu[5]+uu[7])*0.5;
        st[4]=(uu[6]+uu[2])*0.5;
        st[5]=(uu[3]+uu[1])*0.5;

        rez=2.0*nu*(st[0]*st[0]+st[1]*st[1]+st[2]*st[2]+2.0*(st[3]*st[3]+st[4]*st[4]+st[5]*st[5]));

        return rez;
    }
};


class BrinkmanPowerDissipation:public mfem::NonlinearFormIntegrator
{
public:
    BrinkmanPowerDissipation(mfem::GridFunction& velocity,
                             mfem::FluidInterpolationCoefficient& brm_):
                             vel(velocity),brm(brm_)
    {
    }

    virtual ~BrinkmanPowerDissipation()
    {
    }

    virtual
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr,
                               const Vector &elfun, Vector &elvect)
    {
         const int dof = el.GetDof();
         const int dim = el.GetDim();
         elvect.SetSize(dof);
         elvect=0.0;

         int vo = vel.FESpace()->GetOrder(0);
         const mfem::IntegrationRule *ir = nullptr;
         int order= 2 * vo  + Tr.OrderGrad(&el);
         ir=&IntRules.Get(Tr.GetGeometryType(),order);

         mfem::Vector sh(dof);
         mfem::Vector vv(dim);

         double gbpenal; //gradient of the Brinkmann penalization
         double dp;
         double w;

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            w=Tr.Weight();
            w = ip.weight * w;

            gbpenal=brm.GradEval(Tr,ip);

            vel.GetVectorValue(Tr,ip,vv);
            dp=vv*vv;
            el.CalcPhysShape(Tr,sh);

            for(int i=0;i<dof;i++)
            {
                elvect[i]=elvect[i]+w*sh[i]*gbpenal*dp;
            }

         }

    }


    void SetBrinkmanCoefficient(mfem::FluidInterpolationCoefficient& brm_)
    {
        brm=brm_;
    }

    void SetVelocity(mfem::GridFunction& vel_)
    {
        vel=vel_;
    }

private:
    mfem::FluidInterpolationCoefficient& brm;
    mfem::GridFunction& vel;
};

/// The class evaluates the gradient of the objective with respect
/// to velocity multiplied with  the difference between the HOR
/// and LOR velocities.
class PowerDissipationIntegratorEl:public mfem::NonlinearFormIntegrator
{
public:
  PowerDissipationIntegratorEl(mfem::Coefficient* mu_=nullptr,
                                mfem::Coefficient* brm_=nullptr,
                                int velocity_order=3):mu(mu_),brm(brm_)
  {

      lorvel=nullptr;
      horvel=nullptr;
      vo=velocity_order;

  }
  virtual ~PowerDissipationIntegratorEl(){ }

  void SetParameters(mfem::Coefficient* mu_,mfem::Coefficient* brm_)
  {
      mu=mu_;
      brm=brm_;
  }

  void SetGridFunctions(mfem::GridFunction& lorvel_, mfem::GridFunction& horvel_)
  {
      lorvel=&lorvel_;
      horvel=&horvel_;
  }

  virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                  mfem::ElementTransformation &trans,
                                  const mfem::Vector &elfun)
  {
      double objective=0.0;

      const int dim = el.GetDim();

      const int spaceDim = trans.GetDimension();
      if (dim != spaceDim)
      {
         mfem::mfem_error("PowerDissipationIntegratorEl"
                          " is not defined on manifold meshes");
      }


      int order = 2 * vo + trans.OrderGrad(&el);
      const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector gradl(9); gradl=0.0;
      mfem::Vector gradh(9); gradh=0.0;
      mfem::Vector grado(9); grado=0.0;

      mfem::Vector ldispl(dim); ldispl=0.0;
      mfem::Vector hdispl(dim); hdispl=0.0;

      mfem::DenseMatrix lgrad(dim);
      mfem::DenseMatrix hgrad(dim);

      double w;
      Vector vparam(1); vparam[0]=0.0; //viscosisty
      double brink_penal=0.0;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
          const mfem::IntegrationPoint &ip = ir->IntPoint(i);
          trans.SetIntPoint(&ip);
          w = trans.Weight();
          w = ip.weight * w;


          //compute the displacements/velocities
          lorvel->GetVectorValue(trans,ip,ldispl);
          horvel->GetVectorValue(trans,ip,hdispl);

          lorvel->GetVectorGradient(trans,lgrad);
          horvel->GetVectorGradient(trans,hgrad);

          //set the gradients
          for(int i=0;i<dim;i=i+1){
          for(int j=0;j<dim;j=j+1){
              gradl[j*3+i]=lgrad(j,i);
              gradh[j*3+i]=hgrad(j,i);
          }}


          //compute viscosisty
          if(mu!=nullptr){
                  vparam[0] = mu -> Eval(trans,ip);}

          //compute Brinkman penalization
          if(brm!=nullptr){
                  brink_penal = brm -> Eval(trans,ip);}

          adf.QGrad(vparam,gradl,grado);

          objective = objective + w * (grado*gradh-grado*gradl)*0.5;
          objective = objective + w * brink_penal * (ldispl*hdispl-ldispl*ldispl);

      }
      return objective;

  }

  //elfun is a dummy argument
  virtual
  void AssembleElementVector(const FiniteElement &el,
                             ElementTransformation &trans,
                             const Vector &elfun,
                             Vector &elvect)
  {
      const int dim = el.GetDim();
      const int dof = el.GetDof();

      elvect.SetSize(dof);
      elvect=0.0;

      const int spaceDim = trans.GetDimension();
      if (dim != spaceDim)
      {
         mfem::mfem_error("PowerDissipationIntegratorEl"
                          " is not defined on manifold meshes");
      }


      int order = 2 * vo + trans.OrderGrad(&el);
      const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector gradl(9); gradl=0.0;
      mfem::Vector gradh(9); gradh=0.0;
      mfem::Vector grado(9); grado=0.0;

      mfem::Vector ldispl(dim); ldispl=0.0;
      mfem::Vector hdispl(dim); hdispl=0.0;

      mfem::DenseMatrix lgrad(dim);
      mfem::DenseMatrix hgrad(dim);

      mfem::Vector sh(dof);

      double w;
      Vector vparam(1); vparam[0]=0.0; //viscosisty
      double brink_penal=0.0;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
          const mfem::IntegrationPoint &ip = ir->IntPoint(i);
          trans.SetIntPoint(&ip);
          w = trans.Weight();
          w = ip.weight * w;

          el.CalcPhysShape(trans,sh);


          //compute the displacements/velocities
          lorvel->GetVectorValue(trans,ip,ldispl);
          horvel->GetVectorValue(trans,ip,hdispl);

          lorvel->GetVectorGradient(trans,lgrad);
          horvel->GetVectorGradient(trans,hgrad);

          //set the gradients
          for(int i=0;i<dim;i=i+1){
          for(int j=0;j<dim;j=j+1){
              gradl[j*3+i]=lgrad(j,i);
              gradh[j*3+i]=hgrad(j,i);
          }}


          //compute viscosisty
          if(mu!=nullptr){
                  vparam[0] = mu -> Eval(trans,ip);}

          //compute Brinkman penalization
          if(brm!=nullptr){
                  brink_penal = brm -> Eval(trans,ip);}

          adf.QGrad(vparam,gradl,grado);

          double loc=0.0;
          loc = loc + w * (grado*gradh-grado*gradl)*0.5;
          loc = loc + w * brink_penal * (ldispl*hdispl-ldispl*ldispl);

          for(int jj=0;jj<dof;jj++)
          {
                elvect[jj]=elvect[jj]+sh[jj]*loc;
          }

      }

  }


private:
  mfem::Coefficient* brm;
  mfem::Coefficient* mu;
  mfem::QFunctionAutoDiff<PowerDissipationFunctional,9,1> adf;

  int vo;

  mfem::GridFunction* horvel; //velocity
  mfem::GridFunction* lorvel; //adjoint
};




/// Integrator for the PowerDissipationObjective
class PowerDissipationIntegrator:public mfem::NonlinearFormIntegrator
{
public:
    PowerDissipationIntegrator(mfem::Coefficient* mu_=nullptr,
                               mfem::Coefficient* brm_=nullptr):mu(mu_),brm(brm_)
    {

    }

    virtual ~PowerDissipationIntegrator(){ }

    void SetParameters(mfem::Coefficient* mu_,mfem::Coefficient* brm_)
    {
        mu=mu_;
        brm=brm_;
    }

    virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun)
    {
        double objective=0.0;
        const int ndof = el.GetDof();
        const int ndim = el.GetDim();

        const int spaceDim = trans.GetDimension();
        if (ndim != spaceDim)
        {
           mfem::mfem_error("PowerDissipationIntegrator"
                            " is not defined on manifold meshes");
        }

        Vector uu(elfun.GetData()+0*ndof,ndof);
        Vector vv(elfun.GetData()+1*ndof,ndof);
        Vector ww;
        if(ndim==2)
        {
            ww.SetSize(ndof); ww=0.0;
        }
        else
        {
            ww.SetDataAndSize(elfun.GetData()+2*ndof,ndof);
        }

        int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        mfem::DenseMatrix dshape(ndof,ndim);
        mfem::Vector gradu(9); gradu=0.0;
        mfem::Vector displ(3); displ=0.0;

        double w;
        Vector tmpv;
        Vector vparam(1); vparam[0]=0.0; //viscosisty
        double brink_penal=0.0;

        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            w = ip.weight * w;

            el.CalcPhysDShape(trans,dshape);
            el.CalcPhysShape(trans, shapef);

            tmpv.SetDataAndSize(gradu.GetData()+0,ndim);
            dshape.MultTranspose(uu,tmpv);
            tmpv.SetDataAndSize(gradu.GetData()+3,ndim);
            dshape.MultTranspose(vv,tmpv);
            tmpv.SetDataAndSize(gradu.GetData()+6,ndim);
            dshape.MultTranspose(ww,tmpv);

            //compute the displacements/velocities
            displ[0] = shapef * uu;
            displ[1] = shapef * vv;
            displ[2] = shapef * ww;

            //compute viscosisty
            if(mu!=nullptr){
                    vparam[0] = mu -> Eval(trans,ip);}

            //compute Brinkman penalization
            if(brm!=nullptr){
                    brink_penal = brm -> Eval(trans,ip);}

            //compute the objective contribution
            objective = objective + w * adf.QEval(vparam,gradu);
            objective = objective + w * brink_penal * (displ[0]*displ[0]+displ[1]*displ[1]+displ[2]*displ[2]);
        }
        return objective;

    }

    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const Vector &elfun,
                                       Vector &elvect)
    {
        const int ndof = el.GetDof();
        const int ndim = el.GetDim();

        elvect.SetSize(ndim*ndof);

        const int spaceDim = trans.GetDimension();
        if (ndim != spaceDim)
        {
           mfem::mfem_error("PowerDissipationIntegrator"
                            " is not defined on manifold meshes");
        }

        Vector uu(elfun.GetData()+0*ndof,ndof);
        Vector vv(elfun.GetData()+1*ndof,ndof);

        Vector ru(elvect.GetData()+0*ndof,ndof); ru=0.0;
        Vector rv(elvect.GetData()+1*ndof,ndof); rv=0.0;


        Vector ww;
        Vector rw;
        if(ndim==2)
        {
            ww.SetSize(ndof); ww=0.0;
            rw.SetSize(ndof); rw=0.0;

        }
        else
        {
            ww.SetDataAndSize(elfun.GetData()+2*ndof,ndof);
            rw.SetDataAndSize(elvect.GetData()+2*ndof,ndof); rw=0.0;
        }

        int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector shapef(ndof);
        mfem::DenseMatrix dshape(ndof,ndim);
        mfem::Vector gradu(9); gradu=0.0;
        mfem::Vector grado(9);
        mfem::Vector displ(3); displ=0.0;

        mfem::Vector vparam(1); vparam[0]=0.0;
        double brink_penal; brink_penal=0.0;

        Vector tmpv;
        double w;

        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            w = ip.weight * w;

            el.CalcPhysDShape(trans,dshape);
            el.CalcPhysShape(trans, shapef);

            tmpv.SetDataAndSize(gradu.GetData()+0,ndim);
            dshape.MultTranspose(uu,tmpv);
            tmpv.SetDataAndSize(gradu.GetData()+3,ndim);
            dshape.MultTranspose(vv,tmpv);
            tmpv.SetDataAndSize(gradu.GetData()+6,ndim);
            dshape.MultTranspose(ww,tmpv);

            //compute the displacements
            displ[0] = shapef * uu;
            displ[1] = shapef * vv;
            displ[2] = shapef * ww;

            //compute viscosisty
            if(mu!=nullptr){
                    vparam[0] = mu -> Eval(trans,ip);}

            //compute Brinkman penalization
            if(brm!=nullptr){
                    brink_penal = brm -> Eval(trans,ip);}


            adf.QGrad(vparam,gradu,grado);

            tmpv.SetDataAndSize(grado.GetData()+0,ndim);
            dshape.AddMult_a(w,tmpv,ru);
            tmpv.SetDataAndSize(grado.GetData()+3,ndim);
            dshape.AddMult_a(w,tmpv,rv);
            tmpv.SetDataAndSize(grado.GetData()+6,ndim);
            dshape.AddMult_a(w,tmpv,rw);

            ru.Add(2.0*w*displ[0]*brink_penal,shapef);
            rv.Add(2.0*w*displ[1]*brink_penal,shapef);
            rw.Add(2.0*w*displ[2]*brink_penal,shapef);
        }
    }

    virtual void AssembleElementGrad(const FiniteElement &el,
                                     ElementTransformation &Tr,
                                     const Vector &elfun,
                                     DenseMatrix &elmat)
    {

    }


private:
    mfem::Coefficient* brm;
    mfem::Coefficient* mu;
    mfem::QFunctionAutoDiff<PowerDissipationFunctional,9,1> adf;


};

/// Integrator for the error of the PowerDissipationFunctional
class PowerDissipationIntegratorErr:public LinearFormIntegrator
{
public:
    PowerDissipationIntegratorErr(mfem::Coefficient* mu_=nullptr,
                                  mfem::Coefficient* brm_=nullptr):
                                     mu(mu_),brm(brm_)
    {
        lorveloc=nullptr;
        horveloc=nullptr;
    }

    void SetParameters(mfem::Coefficient* mu_,mfem::Coefficient* brm_)
    {
        mu=mu_;
        brm=brm_;
    }

    void SetGridFunctions(mfem::GridFunction& lorvel_, mfem::GridFunction& horvel_)
    {
        lorveloc=&lorvel_;
        horveloc=&horvel_;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el,
                                ElementTransformation &trans,
                                Vector &elvect)
    {
        if(horveloc==nullptr){
            MFEM_ABORT("Please, set the grid functions in PowerDissipationIntegratorErr");
        }

        if(lorveloc==nullptr){
            MFEM_ABORT("Please, set the grid functions in PowerDissipationIntegratorErr");
        }

        const int dim = el.GetDim();
        const int dof = el.GetDof();

        elvect.SetSize(dof);
        elvect=0.0;

        //get the order of the velocity space
        int vo=horveloc->FESpace()->GetOrder(0);

        int order = 3 * vo + trans.OrderGrad(&el);
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        mfem::Vector gradl(9); gradl=0.0;
        mfem::Vector gradh(9); gradh=0.0;
        mfem::Vector grado(9); grado=0.0;

        //velocities
        mfem::Vector ldispl(dim); ldispl=0.0;
        mfem::Vector hdispl(dim); hdispl=0.0;

        mfem::DenseMatrix lgrad(dim);
        mfem::DenseMatrix hgrad(dim);

        //shape function
        mfem::Vector sh(dof);

        double w;
        Vector vparam(1); vparam[0]=0.0; //viscosisty
        double brink_penal=0.0;

        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            w = trans.Weight();
            w = ip.weight * w;

            el.CalcPhysShape(trans,sh);


            //compute the displacements/velocities
            lorveloc->GetVectorValue(trans,ip,ldispl);
            horveloc->GetVectorValue(trans,ip,hdispl);

            lorveloc->GetVectorGradient(trans,lgrad);
            horveloc->GetVectorGradient(trans,hgrad);

            //set the gradients
            for(int ii=0;ii<dim;ii=ii+1){
            for(int jj=0;jj<dim;jj=jj+1){
                gradl[jj*3+ii]=lgrad(jj,ii);
                gradh[jj*3+ii]=hgrad(jj,ii);
            }}


            //compute viscosisty
            if(mu!=nullptr){
                    vparam[0] = mu -> Eval(trans,ip);}

            //compute Brinkman penalization
            if(brm!=nullptr){
                    brink_penal = brm -> Eval(trans,ip);}

            adf.QGrad(vparam,gradl,grado);

            double loc=0.0;
            loc = loc + w * (grado*gradh-grado*gradl)*0.5;
            loc = loc + w * brink_penal * (ldispl*hdispl-ldispl*ldispl);

            for(int jj=0;jj<dof;jj++)
            {
                  elvect[jj]=elvect[jj]+sh[jj]*loc;
            }
        }

    }

private:
    mfem::GridFunction* lorveloc;//LOR solution
    mfem::GridFunction* horveloc;//HOR solution

    int vo;//HOR velocity order

    mfem::Coefficient* brm;
    mfem::Coefficient* mu;
    mfem::QFunctionAutoDiff<PowerDissipationFunctional,9,1> adf;
};



class StokesSolver
{
public:
    StokesSolver(mfem::ParMesh* pmesh_, int vorder=2)
    {
        if(vorder<2){vorder=2;}
        int porder=vorder-1;
        pmesh=pmesh_;

        int dim=pmesh->Dimension();

        vfec=new H1_FECollection(vorder,dim);
        pfec=new H1_FECollection(porder,dim);


        vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);
        pfes=new mfem::ParFiniteElementSpace(pmesh,pfec);

        sfes.Append(vfes);
        sfes.Append(pfes);

        bpenal=nullptr;
        load=nullptr;
        viscosity=nullptr;

        mfem::Array<mfem::ParFiniteElementSpace*> pf;
        pf.Append(vfes);
        pf.Append(pfes);

        nf=new mfem::ParBlockNonlinearForm(pf);
        nfin=nullptr;

        rhs.Update(nf->GetBlockTrueOffsets()); rhs=0.0;
        sol.Update(nf->GetBlockTrueOffsets()); sol=0.0;
        adj.Update(nf->GetBlockTrueOffsets()); adj=0.0;
        tmv.Update(nf->GetBlockTrueOffsets()); tmv=0.0;

        fvelocity.SetSpace(vfes); fvelocity=0.0;
        fpressure.SetSpace(pfes); fpressure=0.0;
        avelocity.SetSpace(vfes); avelocity=0.0;
        apressure.SetSpace(pfes); apressure=0.0;


        pmat=nullptr;
        prec=nullptr;
        psol=nullptr;

        dfes=nullptr;
        ltopopt.fcoef=nullptr;
        ltopopt.tcoef=nullptr;
        SetDesignParameters();

        SetSolver();

        efec=nullptr;
        efes=nullptr;

        smfem.A=nullptr;
        smfem.blPr=nullptr;
        smfem.invA=nullptr;
        smfem.invS=nullptr;
        smfem.S=nullptr;
        smfem.M=nullptr;

    }

    ~StokesSolver()
    {
        delete ltopopt.fcoef;
        delete ltopopt.tcoef;

        delete efes;
        delete efec;

        delete psol;
        delete prec;
        delete pmat;

        delete nf;

        delete vfes;
        delete pfes;
        delete vfec;
        delete pfec;

        delete smfem.blPr;
        delete smfem.invA;
        delete smfem.invS;
        delete smfem.S;
    }


    //this method should be called after mesh refinement
    void Update()
    {
        //Update the design
        fvelocity.SetFromTrueDofs(sol.GetBlock(0));
        fpressure.SetFromTrueDofs(sol.GetBlock(1));
        avelocity.SetFromTrueDofs(adj.GetBlock(0));
        apressure.SetFromTrueDofs(adj.GetBlock(1));


        //update all FES
        vfes->Update();
        pfes->Update();
        if(dfes!=nullptr){
            dfes->Update();
            density.Update();
        }
        if(efes!=nullptr)
        {
            efes->Update();
        }

        fvelocity.Update();
        fpressure.Update();
        avelocity.Update();
        apressure.Update();

        delete psol; psol=nullptr;
        delete prec; prec=nullptr;
        delete pmat; pmat=nullptr;

        delete smfem.blPr; smfem.blPr=nullptr;
        delete smfem.invA; smfem.invA=nullptr;
        delete smfem.invS; smfem.invS=nullptr;
        delete smfem.S; smfem.S=nullptr;

        delete nf;

        mfem::Array<mfem::ParFiniteElementSpace*> pf;
        pf.Append(vfes);
        pf.Append(pfes);

        nf=new mfem::ParBlockNonlinearForm(pf);
        nfin=nullptr;

        rhs.SetSize(1);
        sol.SetSize(1);
        adj.SetSize(1);
        tmv.SetSize(1);
        rhs.Update(nf->GetBlockTrueOffsets()); rhs=0.0;
        sol.Update(nf->GetBlockTrueOffsets());
        adj.Update(nf->GetBlockTrueOffsets());
        tmv.Update(nf->GetBlockTrueOffsets()); tmv=0.0;


        fvelocity.GetTrueDofs(sol.GetBlock(0));
        fpressure.GetTrueDofs(sol.GetBlock(1));
        avelocity.GetTrueDofs(adj.GetBlock(0));
        apressure.GetTrueDofs(adj.GetBlock(1));

    }

    void SetViscosity(mfem::Coefficient& coef_)
    {
        viscosity=&coef_;
    }

    // Set the penalization field to coef_.
    void SetBrinkmanPenal(mfem::Coefficient& coef_)
    {
        bpenal=&coef_;
    }

    void SetVolForces(mfem::VectorCoefficient& load_)
    {
        load=&load_;
    }

    mfem::VectorCoefficient* GetVolForces()
    {
        return load;
    }

    void SetSolver(double rtol=1e-8, double atol=1e-12,int miter=1000, int prt_level=1)
    {
        rel_tol=rtol;
        abs_tol=atol;
        max_iter=miter;
        print_level=prt_level;
    }


    /// Solves the forward problem.
    void FSolve();

    void FSolveN();

    /// Solves the adjoint with the provided rhs.
    void ASolve(mfem::BlockVector& rhs);

    void ASolveN(mfem::BlockVector& rhs);

    /// Return adj*d(residual(sol))/d(design). The dimension
    /// of the vector grad is the save as the dimension of the
    /// true design vector.
    void GradD(mfem::Vector& grad);

    mfem::ParGridFunction& GetVelocity()
    {
        fvelocity.SetFromTrueDofs(sol.GetBlock(0));
        return fvelocity;
    }

    mfem::ParGridFunction& GetPressure()
    {
        fpressure.SetFromTrueDofs(sol.GetBlock(1));
        return fpressure;
    }

    mfem::ParGridFunction& GetDesign()
    {
        return density;
    }

    mfem::ParGridFunction& GetAVelocity()
    {
        avelocity.SetFromTrueDofs(adj.GetBlock(0));
        return avelocity;
    }

    mfem::ParGridFunction& GetAPressure()
    {
        apressure.SetFromTrueDofs(adj.GetBlock(1));
        return apressure;
    }

    void AddVelocityBC(int id, int dir, double val)
    {
        if(dir==0){
            bcx[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcx[id]);
        }
        if(dir==1){
            bcy[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcy[id]);

        }
        if(dir==2){
            bcz[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,dir,bcz[id]);
        }
        if(dir==4){
            bcx[id]=mfem::ConstantCoefficient(val);
            bcy[id]=mfem::ConstantCoefficient(val);
            bcz[id]=mfem::ConstantCoefficient(val);
            AddVelocityBC(id,0,bcx[id]);
            AddVelocityBC(id,1,bcy[id]);
            AddVelocityBC(id,2,bcz[id]);
        }
    }

    void AddVelocityBC(int id, int dir, mfem::Coefficient& val)
    {
        if(dir==0){ bccx[id]=&val; }
        if(dir==1){ bccy[id]=&val; }
        if(dir==2){ bccz[id]=&val; }
        if(dir==4){ bccx[id]=&val; bccy[id]=&val; bccz[id]=&val;}
        if(pmesh->Dimension()==2)
        {
            bccz.clear();
        }
    }


    mfem::BlockVector& GetSol(){return sol;}
    mfem::BlockVector& GetAdj(){return adj;}

    mfem::ParFiniteElementSpace* GetVelocityFES(){return vfes;}
    mfem::ParFiniteElementSpace* GetPressureFES(){return pfes;}
    mfem::ParFiniteElementSpace* GetDesignFES(){return dfes;}

    mfem::ParFiniteElementSpace* GetErrorFES(){
        if(efes==nullptr)
        {
            int dim=pmesh->Dimension();
            efec=new mfem::L2_FECollection(0,dim);
            efes=new mfem::ParFiniteElementSpace(pmesh,efec,1);
        }
        return efes;
    }

    ///Get state spaces
    mfem::Array<mfem::ParFiniteElementSpace*>& GetStateFES()
    {
        return sfes;
    }

    void SetDesignSpace(mfem::ParFiniteElementSpace* dfes_)
    {
        dfes=dfes_;
        density.SetSpace(dfes);
        density=0.0;

        if(ltopopt.fcoef!=nullptr)
        {
            ltopopt.fcoef->SetGridFunction(density);
        }else{
            ltopopt.fcoef=new FluidInterpolationCoefficient(ltopopt.eta,ltopopt.beta,
                                                        ltopopt.q,ltopopt.lambda,density);
        }
        bpenal=ltopopt.fcoef;
    }

    // Sets the design field using true vector designvec
    // and set the penalization field
    void SetDesign(mfem::Vector& designvec)
    {
        density.SetFromTrueDofs(designvec);
        if(ltopopt.fcoef==nullptr)
        {
            ltopopt.fcoef=new FluidInterpolationCoefficient(ltopopt.eta,ltopopt.beta,
                                                            ltopopt.q,ltopopt.lambda,density);
        }

        ltopopt.fcoef->SetGridFunction(density);
        bpenal=ltopopt.fcoef;
    }

    void SetDesignParameters(double eta_=0.5, double beta_=8.0, double q_=1, double lambda_=100)
    {
        ltopopt.eta=eta_;
        ltopopt.beta=beta_;
        ltopopt.lambda=lambda_;
        ltopopt.q=q_;

        if(ltopopt.fcoef!=nullptr)
        {
            ltopopt.fcoef->SetParameters(ltopopt.eta,ltopopt.beta,
                                         ltopopt.q,ltopopt.lambda);
        }
    }

    void SetTargetDesignParameters(double eta_=0.5, double beta_=8.0, double q_=1, double lambda_=10000)
    {
        if(ltopopt.tcoef==nullptr)
        {
            ltopopt.tcoef=new mfem::FluidInterpolationCoefficient();
        }
        ltopopt.tcoef->SetParameters(eta_,beta_,q_,lambda_);
    }

    mfem::Coefficient* GetViscosity(){ return viscosity;}

    mfem::FluidInterpolationCoefficient* GetBrinkmanPenal(){
        if(ltopopt.fcoef==nullptr)
        {
            SetDesignParameters();
        }
        ltopopt.fcoef->SetGridFunction(density);
        return ltopopt.fcoef;
    }

    mfem::FluidInterpolationCoefficient* GetTargetBrinkmanPenal(){
        if(ltopopt.tcoef==nullptr)
        {
            SetTargetDesignParameters();
        }

        ltopopt.tcoef->SetGridFunction(density);
        return ltopopt.tcoef;
    }

    /// Returns vector with size equal to the number of elements located on the process.
    /// The coefficient represents the exact (the target) model. Every element of the vector
    /// represents the elemental contribution to the model error.
    double ModelErrors(mfem::GridFunction& el_errors);

    /// Returns vector with size equal to the number of elements located on the process.
    /// Every element of the vector represents the elemental contribution to the
    /// discretization error
    /// void DiscretizationErrors(mfem::Vector& el_errors);


    mfem::ParMesh* GetMesh(){return pmesh;}

    /// Transfer BC and viscosity and load coefficients to nsolver.
    /// The Brinkman penalization is not transfered.
    void TransferData(StokesSolver* nsolver);


private:



    double mu;
    double alpha;
    mfem::Coefficient* viscosity;
    mfem::Coefficient* bpenal;
    mfem::VectorCoefficient* load;

    mfem::ParMesh* pmesh;

    mfem::ParFiniteElementSpace* vfes;
    mfem::ParFiniteElementSpace* pfes;
    mfem::FiniteElementCollection* vfec;
    mfem::FiniteElementCollection* pfec;
    mfem::Array<mfem::ParFiniteElementSpace*> sfes;

    // finite element space for error estimation
    mfem::FiniteElementCollection* efec;
    mfem::ParFiniteElementSpace* efes;

    // boundary conditions
    std::map<int, mfem::ConstantCoefficient> bcx;
    std::map<int, mfem::ConstantCoefficient> bcy;
    std::map<int, mfem::ConstantCoefficient> bcz;

    std::map<int, mfem::Coefficient*> bccx;
    std::map<int, mfem::Coefficient*> bccy;
    std::map<int, mfem::Coefficient*> bccz;

    mfem::Array<int> ess_tdofv;


    //mfem::BlockNonlinearFormIntegrator* nfin;
    mfem::StokesIntegratorTH* nfin;
    mfem::ParBlockNonlinearForm* nf;

    mfem::BlockVector rhs;
    mfem::BlockVector sol;
    mfem::BlockVector adj;
    mfem::BlockVector tmv;// temp vector

    mfem::ParGridFunction fvelocity;
    mfem::ParGridFunction avelocity;
    mfem::ParGridFunction fpressure;
    mfem::ParGridFunction apressure;

    /// The PETSc objects are allocated once the problem is
    /// assembled. They are utilized in computing the adjoint
    /// solutions.
    mfem::PetscParMatrix* pmat;
    mfem::PetscPreconditioner* prec;
    mfem::PetscLinearSolver*   psol;

    struct
    {
        mfem::BlockOperator* A;
        mfem::Solver *invA;
        mfem::Solver *invS;
        mfem::HypreParMatrix *S;
        mfem::BlockDiagonalPreconditioner *blPr;
        mfem::Array<int> block_trueOffsets;
        mfem::HypreParMatrix *M;

    } smfem;

    double abs_tol;
    double rel_tol;
    int print_level;
    int max_iter;

    mfem::ParFiniteElementSpace* dfes; //design space
    mfem::ParGridFunction density;

    struct{
      double eta;
      double beta;
      double lambda;
      double q;
      mfem::FluidInterpolationCoefficient* fcoef;
      mfem::FluidInterpolationCoefficient* tcoef;//target coefficient
    } ltopopt;

};

class VolumeQoI
{
public:
    VolumeQoI(StokesSolver* solver_, double vol_=0.5):one(1.0)
    {
        solver=solver_;
        dfes=solver->GetDesignFES();
        lf=new mfem::ParLinearForm(dfes);
        //compute the total volume
        mfem::ParGridFunction gfone(dfes);
        gfone.ProjectCoefficient(one);
        lf=new mfem::ParLinearForm(dfes);
        lf->AddDomainIntegrator(new  mfem::DomainLFIntegrator(one, dfes->GetElementOrder(0)));
        lf->Assemble();
        tot_vol=(*lf)(gfone);
        tot_vol=tot_vol;
        vol=vol_;
    }

    ~VolumeQoI()
    {
        delete lf;
    }

    double Eval()
    {
        mfem::ParGridFunction& dens=solver->GetDesign();
        double cur_vol=(*lf)(dens);

        mfem::ParMesh* pmesh=solver->GetMesh();
        int rank;
        MPI_Comm_rank(pmesh->GetComm(),&rank);
        if(rank==0)
        {
            std::cout<<"current vol="<<cur_vol<<" total vol"<<tot_vol<<" %="<<cur_vol/tot_vol<<std::endl;
        }

        return cur_vol/(tot_vol*vol)-1.0;
    }

    /// Input: true design vector
    double Eval(mfem::Vector& design_)
    {
        design.SetSpace(dfes);
        design.SetFromTrueDofs(design_);
        double cur_vol=(*lf)(design);
        return cur_vol/(tot_vol*vol)-1.0;
    }

    void Grad(mfem::Vector& grad)
    {
        lf->ParallelAssemble(grad);
        grad/=(tot_vol*vol);
    }
private:
    mfem::StokesSolver* solver;
    mfem::ParFiniteElementSpace* dfes;
    mfem::ParLinearForm* lf;
    mfem::ConstantCoefficient one;
    mfem::ParGridFunction design;
    double tot_vol;
    double vol;

};



class VelocityIntErr:public mfem::LinearFormIntegrator
{
public:
    VelocityIntErr()
    {

    }

    virtual
    ~VelocityIntErr()
    {

    }

    void SetLORFields(mfem::GridFunction& vel)
    {
        lorVelocity=&vel;
    }

    void SetHORFields(mfem::GridFunction& vel)
    {
        horVelocity=&vel;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el,
                                ElementTransformation &Tr,
                                Vector &elvect) override
    {

    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el,
                                FaceElementTransformations &Tr,
                                Vector &elvect) override
    {
        const int dim = el.GetDim();
        const int dof = el.GetDof();
        double nor_data[3];

        Vector lorvel(dim);
        Vector horvel(dim);

        mfem::Vector nor(nor_data,dim);

        int vo=horVelocity->FESpace()->GetOrder(Tr.Elem1->ElementNo);
        int order = 2 * vo + Tr.Elem1->OrderW();
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(Tr.GetGeometryType(), order);

        Vector shape(dof);
        elvect.SetSize(dof);
        elvect=0.0;
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);

           // Set the integration point in the face and the neighboring element
           Tr.SetAllIntPoints(&ip);

           // Access the neighboring element's integration point
           const IntegrationPoint &eip = Tr.GetElement1IntPoint();
           el.CalcShape(eip, shape);

           CalcOrtho(Tr.Jacobian(), nor);

           lorVelocity->GetVectorValue(*Tr.Elem1,eip, lorvel);
           horVelocity->GetVectorValue(*Tr.Elem1,eip, horvel);

           elvect.Add(0.5*ip.weight*(horvel*nor), shape);
           elvect.Add(-0.5*ip.weight*(lorvel*nor), shape);
        }
    }


private:
   mfem::GridFunction* lorVelocity;
   mfem::GridFunction* horVelocity;
};

class VelocityIntQoI
{
public:
    VelocityIntQoI(StokesSolver* solver_, int face_id_):cf(1.0),arhs(solver_->GetSol())
    {
        solver=solver_;
        face_id=face_id_;
        lf=new mfem::ParLinearForm(solver->GetVelocityFES());
        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=0;
        bdr_att[face_id-1]=1;
        lf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(cf),bdr_att);
        lf->Assemble();
    }

    ~VelocityIntQoI()
    {
        delete lf;
    }

    double Eval()
    {
        mfem::ParGridFunction& vel=solver->GetVelocity();
        return (*lf)(vel);
    }

    void Grad(mfem::Vector& grad)
    {
        arhs=0.0;
        lf->ParallelAssemble(arhs.GetBlock(0));
        solver->ASolve(arhs);
        solver->GradD(grad);
    }

    double ModelError(mfem::GridFunction& moderr)
    {
        arhs=0.0;
        lf->ParallelAssemble(arhs.GetBlock(0));
        solver->ASolve(arhs);
        return solver->ModelErrors(moderr);
    }

    double DiscretizationError(mfem::GridFunction& diserr)
    {

        //compute the adjoint
        {
            arhs=0.0;
            lf->ParallelAssemble(arhs.GetBlock(0));
            solver->ASolve(arhs);
        }

        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=0;
        bdr_att[face_id-1]=1;

        //create a new Stokes solver
        mfem::StokesSolver* hsolver=new mfem::StokesSolver(solver->GetMesh(),
                                                           (solver->GetVelocityFES()->GetOrder(0))+1 );
        solver->TransferData(hsolver);
        //transfer the forward and the adjoint solutions
        {
            mfem::ParGridFunction& vel=hsolver->GetVelocity();
            vel.ProjectGridFunction(solver->GetVelocity());
            vel.GetTrueDofs(hsolver->GetSol().GetBlock(0));

            mfem::ParGridFunction& pre=hsolver->GetPressure();
            pre.ProjectGridFunction(solver->GetPressure());
            pre.GetTrueDofs(hsolver->GetSol().GetBlock(1));

            vel=hsolver->GetAVelocity();
            vel.ProjectGridFunction(solver->GetAVelocity());
            vel.GetTrueDofs(hsolver->GetAdj().GetBlock(0));

            pre=hsolver->GetAPressure();
            pre.ProjectGridFunction(solver->GetAPressure());
            pre.GetTrueDofs(hsolver->GetAdj().GetBlock(1));
        }

        MPI_Comm lcomm;
        lcomm=solver->GetMesh()->GetComm();
        int myrank;
        MPI_Comm_rank(lcomm,&myrank);

        if(myrank==0){std::cout<<"Solution HSolver"<<std::endl;}

        hsolver->FSolve(); //Improved forward solution

        //Improved adjoint solution
        VelocityIntQoI* hqoi=new VelocityIntQoI(hsolver,face_id);
        {
            mfem::BlockVector larhs(hsolver->GetSol());
            larhs=0.0;
            hqoi->lf->ParallelAssemble(larhs.GetBlock(0));
            hsolver->ASolve(larhs);
        }

        double hval=hqoi->Eval();
        if(myrank==0){std::cout<<"HQoi="<<hval<<std::endl;}

        delete hqoi;

        diserr.SetSpace(solver->GetErrorFES());
        diserr=0.0;

        {
            mfem::ParLinearForm* lf=new mfem::ParLinearForm(solver->GetErrorFES());

            StokesResidualIntegrator* rint=
                    new StokesResidualIntegrator(solver->GetViscosity(), solver->GetBrinkmanPenal(),
                                                 solver->GetVolForces());

            rint->SetLORFields(solver->GetVelocity(),solver->GetPressure(),
                               solver->GetAVelocity(), solver->GetAPressure());
            rint->SetHORFields(hsolver->GetVelocity(),hsolver->GetPressure(),
                               hsolver->GetAVelocity(),hsolver->GetAPressure());

            lf->AddDomainIntegrator(rint);

            VelocityIntErr* eint=new VelocityIntErr();
            eint->SetLORFields(solver->GetVelocity());
            eint->SetHORFields(hsolver->GetVelocity());
            lf->AddBdrFaceIntegrator(eint,bdr_att);

            lf->Assemble();

            lf->ParallelAssemble(diserr);
            delete lf;
        }

        double locres=0.0;
        for(int i=0;i<diserr.Size();i++)
        {
            locres=locres+fabs(diserr[i]);
        }

        double totres=0.0;

        MPI_Allreduce(&locres, &totres, 1, MPI_DOUBLE,MPI_SUM,lcomm);

        delete hsolver;

        return totres;
    }

    //This method should be called after mesh refinement/de-refinement
    void Update()
    {
        delete lf;
        lf=new mfem::ParLinearForm(solver->GetVelocityFES());
        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=0;
        bdr_att[face_id-1]=1;
        lf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(cf),bdr_att);
        lf->Assemble();


        mfem::Array<int> bOffsets(solver->GetSol().NumBlocks()+1);
        bOffsets[0]=0;
        for(int i=0;i<solver->GetSol().NumBlocks();i++)
        {
            bOffsets[i+1]=bOffsets[i]+solver->GetSol().BlockSize(i);
        }
        arhs.Update(bOffsets);
    }

private:
    StokesSolver* solver;
    int face_id;
    mfem::ParLinearForm* lf;
    mfem::ConstantCoefficient cf;
    mfem::BlockVector arhs;
};


class AveragePressureDropErr:public mfem::LinearFormIntegrator
{
public:
    AveragePressureDropErr()
    {
        lorVelocity=nullptr;
        lorPressure=nullptr;
        horVelocity=nullptr;
        horPressure=nullptr;
    }

    ~AveragePressureDropErr()
    {

    }

    void SetLORFields(mfem::GridFunction& vel, mfem::GridFunction& press)
    {
        lorVelocity=&vel;
        lorPressure=&press;
    }

    void SetHORFields(mfem::GridFunction& vel, mfem::GridFunction& press)
    {
        horVelocity=&vel;
        horPressure=&press;
    }

    virtual
    void AssembleRHSElementVect(const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect) override
    {
        const int dim = el.GetDim();
        const int dof = el.GetDof();
        double nor_data[3];

        Vector lorvel(dim);
        Vector horvel(dim);

        double lorp, horp;
        mfem::Vector nor(nor_data,dim);

        int vo=horVelocity->FESpace()->GetOrder(Tr.Elem1->ElementNo);
        int order = 2 * vo + Tr.Elem1->OrderW();
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(Tr.GetGeometryType(), order);

        Vector shape(dof);
        elvect.SetSize(dof);
        elvect=0.0;

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
           const IntegrationPoint &ip = ir->IntPoint(p);

           // Set the integration point in the face and the neighboring element
           Tr.SetAllIntPoints(&ip);

           // Access the neighboring element's integration point
           const IntegrationPoint &eip = Tr.GetElement1IntPoint();
           el.CalcShape(eip, shape);

           CalcOrtho(Tr.Jacobian(), nor);

           lorVelocity->GetVectorValue(*Tr.Elem1,eip, lorvel);
           horVelocity->GetVectorValue(*Tr.Elem1,eip, horvel);

           lorp=lorPressure->GetValue(*Tr.Elem1,eip);
           horp=horPressure->GetValue(*Tr.Elem1,eip);

           elvect.Add(-0.5*ip.weight*(lorvel*nor)*(horp-lorp), shape);
           elvect.Add(-0.5*ip.weight*(horvel*nor)*lorp, shape);
           elvect.Add(+0.5*ip.weight*(lorvel*nor)*lorp, shape);
        }
    }



    virtual
    void AssembleRHSElementVect(const FiniteElement &el,
                                ElementTransformation &Tr,
                                Vector &elvect) override
    {
        const int dim = el.GetDim()+1;
        const int dof = el.GetDof();
        Vector nor(dim);
        Vector lorvel(dim);
        Vector horvel(dim);
        Vector shape;
        shape.SetSize(dof);
        elvect.SetSize(dof);
        elvect = 0.0;

        double lorp, horp;

        int vo=horVelocity->FESpace()->GetOrder(Tr.ElementNo);
        int order = 2 * vo + Tr.OrderGrad(&el);
        const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), order);

        // set the integration rule

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
           const IntegrationPoint &ip = ir->IntPoint(i);
           Tr.SetIntPoint(&ip);
           CalcOrtho(Tr.Jacobian(), nor);

           lorVelocity->GetVectorValue(Tr,ip, lorvel);
           horVelocity->GetVectorValue(Tr,ip, horvel);

           lorp=lorPressure->GetValue(Tr,ip);
           horp=horPressure->GetValue(Tr,ip);

           el.CalcShape(ip, shape);

           elvect.Add(-0.5*ip.weight*(lorvel*nor)*(horp-lorp), shape);
           elvect.Add(-0.5*ip.weight*(horvel*nor)*lorp, shape);
           elvect.Add(+0.5*ip.weight*(lorvel*nor)*lorp, shape);
        }

    }


private:
   mfem::GridFunction* lorVelocity;
   mfem::GridFunction* horVelocity;
   mfem::GridFunction* lorPressure;
   mfem::GridFunction* horPressure;

};


class AveragePressureDropQoI
{
public:
    AveragePressureDropQoI(StokesSolver* solver_):arhs(solver_->GetSol())
    {
        MFEM_ASSERT(solver_!=nullptr,
                    "BndrPowerDissipationQoI: The pointer to the solver should be different than nullptr!");

        solver=solver_;

        //define finite element spaces for the error estimators
        mfem::ParMesh* pmesh=solver->GetMesh();
        int dim=pmesh->Dimension();
    }

    ~AveragePressureDropQoI()
    {
    }

    double Eval()
    {

        mfem::ParLinearForm* lf=new mfem::ParLinearForm(solver->GetVelocityFES());
        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=1;

        mfem::GridFunctionCoefficient press(&(solver->GetPressure()));
        lf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(press,1,2,0),bdr_att);
        lf->Assemble();
        double rez=-(*lf)(solver->GetVelocity());
        delete lf;
        return rez;;
    }

    void Grad(mfem::Vector& grad)
    {
        arhs=0.0;

        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=1;

        //pressure
        {
            mfem::ParLinearForm* plf=new mfem::ParLinearForm(solver->GetPressureFES());
            mfem::VectorGridFunctionCoefficient vcf(&(solver->GetVelocity()));
            plf->AddBoundaryIntegrator(
                 new mfem::BoundaryNormalLFIntegrator(vcf,2,2),bdr_att);
            plf->Assemble();
            plf->ParallelAssemble(arhs.GetBlock(1));
            delete plf;
        }

        //velocity
        {
            mfem::ParLinearForm* vlf=new mfem::ParLinearForm(solver->GetVelocityFES());
            mfem::GridFunctionCoefficient press(&(solver->GetPressure()));
            vlf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(press,1,2,0),bdr_att);
            vlf->Assemble();
            vlf->ParallelAssemble(arhs.GetBlock(0));
            delete vlf;
        }

        arhs.Neg();
        solver->ASolve(arhs);
        solver->GradD(grad);
    }

    double ModelError(mfem::GridFunction& moderr)
    {
        arhs=0.0;

        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=1;

        //pressure
        {
            mfem::ParLinearForm* plf=new mfem::ParLinearForm(solver->GetPressureFES());
            mfem::VectorGridFunctionCoefficient vcf(&(solver->GetVelocity()));
            plf->AddBoundaryIntegrator(
                 new mfem::BoundaryNormalLFIntegrator(vcf,2,2),bdr_att);
            plf->Assemble();
            plf->ParallelAssemble(arhs.GetBlock(1));
            delete plf;
        }

        //velocity
        {
            mfem::ParLinearForm* vlf=new mfem::ParLinearForm(solver->GetVelocityFES());
            mfem::GridFunctionCoefficient press(&(solver->GetPressure()));
            vlf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(press,1,2,0),bdr_att);
            vlf->Assemble();
            vlf->ParallelAssemble(arhs.GetBlock(0));
            delete vlf;
        }

        arhs.Neg();
        //compute the adjoint
        solver->ASolve(arhs);
        return solver->ModelErrors(moderr);

    }

    double DiscretizationError(mfem::GridFunction& diserr)
    {
        mfem::ParMesh* pmesh=solver->GetMesh();
        mfem::Array<int> bdr_att; bdr_att.SetSize(pmesh->bdr_attributes.Size());
        bdr_att=1;

        //compute the adjoint
        {
            arhs=0.0;
            //pressure
            mfem::ParLinearForm* plf=new mfem::ParLinearForm(solver->GetPressureFES());
            mfem::VectorGridFunctionCoefficient vcf(&(solver->GetVelocity()));
            plf->AddBoundaryIntegrator(
                 new mfem::BoundaryNormalLFIntegrator(vcf,2,2),bdr_att);
            plf->Assemble();
            plf->ParallelAssemble(arhs.GetBlock(1));
            delete plf;
            //velocity
            mfem::ParLinearForm* vlf=new mfem::ParLinearForm(solver->GetVelocityFES());
            mfem::GridFunctionCoefficient press(&(solver->GetPressure()));
            vlf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(press,1,2,0),bdr_att);
            vlf->Assemble();
            vlf->ParallelAssemble(arhs.GetBlock(0));
            delete vlf;

            arhs.Neg();
            //compute the adjoint
            solver->ASolve(arhs);
        }

        //create a new Stokes solver
        mfem::StokesSolver* hsolver=new mfem::StokesSolver(solver->GetMesh(),
                                                           (solver->GetVelocityFES()->GetOrder(0))+1 );
        solver->TransferData(hsolver);
        //transfer the forward and the adjoint solutions
        {
            mfem::ParGridFunction& vel=hsolver->GetVelocity();
            vel.ProjectGridFunction(solver->GetVelocity());
            vel.GetTrueDofs(hsolver->GetSol().GetBlock(0));

            mfem::ParGridFunction& pre=hsolver->GetPressure();
            pre.ProjectGridFunction(solver->GetPressure());
            pre.GetTrueDofs(hsolver->GetSol().GetBlock(1));

            vel=hsolver->GetAVelocity();
            vel.ProjectGridFunction(solver->GetAVelocity());
            vel.GetTrueDofs(hsolver->GetAdj().GetBlock(0));

            pre=hsolver->GetAPressure();
            pre.ProjectGridFunction(solver->GetAPressure());
            pre.GetTrueDofs(hsolver->GetAdj().GetBlock(1));
        }

        MPI_Comm lcomm;
        lcomm=solver->GetMesh()->GetComm();
        int myrank;
        MPI_Comm_rank(lcomm,&myrank);

        if(myrank==0){std::cout<<"Solution HSolver"<<std::endl;}

        hsolver->FSolve(); //Improved forward solution

        //Improved adjoint solution
        AveragePressureDropQoI* hqoi=new AveragePressureDropQoI(hsolver);
        {
            mfem::BlockVector larhs(hsolver->GetSol());
            larhs=0.0;
            //pressure
            mfem::ParLinearForm* plf=new mfem::ParLinearForm(hsolver->GetPressureFES());
            mfem::VectorGridFunctionCoefficient vcf(&(hsolver->GetVelocity()));
            plf->AddBoundaryIntegrator(
                 new mfem::BoundaryNormalLFIntegrator(vcf,3,2),bdr_att);
            plf->Assemble();
            plf->ParallelAssemble(larhs.GetBlock(1));
            delete plf;
            //velocity
            mfem::ParLinearForm* vlf=new mfem::ParLinearForm(hsolver->GetVelocityFES());
            mfem::GridFunctionCoefficient press(&(hsolver->GetPressure()));
            vlf->AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(press,2,2,0),bdr_att);
            vlf->Assemble();
            vlf->ParallelAssemble(larhs.GetBlock(0));
            delete vlf;

            larhs.Neg();
            //compute the adjoint
            hsolver->ASolve(larhs);
        }

        double hval=hqoi->Eval();
        if(myrank==0){std::cout<<"HQoi="<<hval<<std::endl;}

        delete hqoi;

        diserr.SetSpace(solver->GetErrorFES());
        diserr=0.0;



        {
            mfem::ParLinearForm* lf=new mfem::ParLinearForm(solver->GetErrorFES());

            StokesResidualIntegrator* rint=
                    new StokesResidualIntegrator(solver->GetViscosity(), solver->GetBrinkmanPenal(),
                                                 solver->GetVolForces());

            rint->SetLORFields(solver->GetVelocity(),solver->GetPressure(),
                               solver->GetAVelocity(), solver->GetAPressure());
            rint->SetHORFields(hsolver->GetVelocity(),hsolver->GetPressure(),
                               hsolver->GetAVelocity(),hsolver->GetAPressure());


            lf->AddDomainIntegrator(rint);

            AveragePressureDropErr* eint=new AveragePressureDropErr();
            eint->SetLORFields(solver->GetVelocity(),solver->GetPressure());
            eint->SetHORFields(hsolver->GetVelocity(),hsolver->GetPressure());
            lf->AddBdrFaceIntegrator(eint,bdr_att);

            lf->Assemble();

            lf->ParallelAssemble(diserr);
            delete lf;
        }

        double locres=0.0;
        for(int i=0;i<diserr.Size();i++)
        {
            locres=locres+diserr[i];
        }

        double totres=0.0;

        MPI_Allreduce(&locres, &totres, 1, MPI_DOUBLE,MPI_SUM,lcomm);

        delete hsolver;

        return totres;
    }




private:
    StokesSolver* solver;
    mfem::BlockVector arhs;
};


class PowerDissipationQoI
{
public:
    PowerDissipationQoI(StokesSolver* solver_)
    {
        MFEM_ASSERT(solver_!=nullptr,
                    "PowerDissipationQoI: The pointer to the solver should be different than nullptr!");

        solver=solver_;

        nf=new mfem::ParNonlinearForm(solver->GetVelocityFES());
        intg=new PowerDissipationIntegrator(solver->GetViscosity(),solver->GetBrinkmanPenal());
        nf->AddDomainIntegrator(intg);

        bntg=new BrinkmanPowerDissipation(solver->GetVelocity(),
                                          *(solver->GetBrinkmanPenal()));
        lf=new ParNonlinearForm(solver->GetDesignFES());
        lf->AddDomainIntegrator(bntg);

    }

    ~PowerDissipationQoI()
    {
        delete nf;
        delete lf;
    }

    double Eval()
    {
        mfem::BlockVector& sol=solver->GetSol();
        intg->SetParameters(solver->GetViscosity(),solver->GetBrinkmanPenal());
        double rez = nf->GetEnergy(sol.GetBlock(0)); //we need only the velocities
        return rez;
    }

    void Grad(mfem::Vector& grad)
    {
        //tmpv is used only for providing input to the mult method
        //the values are not used fro anything
        tmpv.SetSize(grad.Size());


        bntg->SetVelocity(solver->GetVelocity());
        bntg->SetBrinkmanCoefficient(*(solver->GetBrinkmanPenal()));
        lf->Mult(tmpv,grad);
    }

    double ModelError(mfem::GridFunction& moderr)
    {
        //gradients with respect to velocity
        mfem::BlockVector arhs(solver->GetSol());
        arhs=0.0;
        intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
        //compute derivative of the objective
        nf->Mult(solver->GetSol().GetBlock(0),arhs.GetBlock(0));
        //compute the adjoint
        solver->ASolve(arhs);
        return solver->ModelErrors(moderr);
    }

    double DiscretizationError(mfem::GridFunction& diserr)
    {
        //compute the adjoint
        {
            mfem::BlockVector arhs(solver->GetSol());
            arhs=0.0;
            intg->SetParameters(solver->GetViscosity(),solver->GetBrinkmanPenal());
            nf->Mult(solver->GetSol().GetBlock(0),arhs.GetBlock(0));
            solver->ASolve(arhs);
        }

        //create a new Stokes solver
        mfem::StokesSolver* hsolver=new mfem::StokesSolver(solver->GetMesh(),
                                                           (solver->GetVelocityFES()->GetOrder(0))+1 );
        solver->TransferData(hsolver);
        //transfer the forward and the adjoint solutions
        {
            mfem::ParGridFunction& vel=hsolver->GetVelocity();
            vel.ProjectGridFunction(solver->GetVelocity());
            vel.GetTrueDofs(hsolver->GetSol().GetBlock(0));

            mfem::ParGridFunction& pre=hsolver->GetPressure();
            pre.ProjectGridFunction(solver->GetPressure());
            pre.GetTrueDofs(hsolver->GetSol().GetBlock(1));

            vel=hsolver->GetAVelocity();
            vel.ProjectGridFunction(solver->GetAVelocity());
            vel.GetTrueDofs(hsolver->GetAdj().GetBlock(0));

            pre=hsolver->GetAPressure();
            pre.ProjectGridFunction(solver->GetAPressure());
            pre.GetTrueDofs(hsolver->GetAdj().GetBlock(1));
        }

        MPI_Comm lcomm;
        lcomm=solver->GetMesh()->GetComm();
        int myrank;
        MPI_Comm_rank(lcomm,&myrank);

        if(myrank==0){std::cout<<"Solution HSolver"<<std::endl;}

        hsolver->FSolve(); //Improved forward solution
        //Improved adjoint solution
        PowerDissipationQoI* hqoi=new PowerDissipationQoI(hsolver);
        {
            mfem::BlockVector arhs(hsolver->GetSol());
            arhs=0.0;
            //the coefficients are the same for solver and hsolver
            hqoi->intg->SetParameters(hsolver->GetViscosity(),hsolver->GetBrinkmanPenal());
            hqoi->nf->Mult(hsolver->GetSol().GetBlock(0),arhs.GetBlock(0));
            hsolver->ASolve(arhs);
        }

        double hval=hqoi->Eval();
        if(myrank==0){std::cout<<"HQoi="<<hval<<std::endl;}

        delete hqoi;


        {
            mfem::ParLinearForm* lf=new mfem::ParLinearForm(solver->GetErrorFES());
            PowerDissipationIntegratorErr* lint=
                    new PowerDissipationIntegratorErr(solver->GetViscosity(),solver->GetBrinkmanPenal());

            lint->SetGridFunctions(solver->GetVelocity(),hsolver->GetVelocity());

            StokesResidualIntegrator* rint=
                    new StokesResidualIntegrator(solver->GetViscosity(), solver->GetBrinkmanPenal(),
                                                 solver->GetVolForces());

            rint->SetLORFields(solver->GetVelocity(),solver->GetPressure(),
                               solver->GetAVelocity(), solver->GetAPressure());
            rint->SetHORFields(hsolver->GetVelocity(),hsolver->GetPressure(),
                               hsolver->GetAVelocity(),hsolver->GetAPressure());


            lf->AddDomainIntegrator(rint);
            lf->AddDomainIntegrator(lint);
            lf->Assemble();

            diserr.SetSpace(solver->GetErrorFES());
            diserr=0.0;

            lf->ParallelAssemble(diserr);
            delete lf;
        }

        double locres=0.0;
        for(int i=0;i<diserr.Size();i++)
        {
            locres=locres+diserr[i];
        }

        double totres=0.0;

        MPI_Allreduce(&locres, &totres, 1, MPI_DOUBLE,MPI_SUM,lcomm);

        delete hsolver;

        return totres;
    }

    void UpdateSolver(StokesSolver* solver_)
    {
        delete nf;
        solver=solver_;
        nf=new mfem::ParNonlinearForm(solver->GetVelocityFES());
        intg=new PowerDissipationIntegrator(solver->GetViscosity(),solver->GetBrinkmanPenal());
        nf->AddDomainIntegrator(intg);

        delete lf;
        bntg=new BrinkmanPowerDissipation(solver->GetVelocity(),
                                          *(solver->GetBrinkmanPenal()));
        lf=new ParNonlinearForm(solver->GetDesignFES());
        lf->AddDomainIntegrator(bntg);
    }


private:
    mfem::StokesSolver* solver;

    mfem::ParNonlinearForm* nf;
    mfem::PowerDissipationIntegrator* intg;

    mfem::BrinkmanPowerDissipation* bntg;
    mfem::ParNonlinearForm* lf;

    mfem::Vector tmpv;
};


/// Modified power dissipation QoI using the target
/// Brinkman penalization instead of the one used in the
/// solution.
class PowerDissipationTGQoI
{
public:
    PowerDissipationTGQoI(StokesSolver* solver_)
    {
        MFEM_ASSERT(solver_!=nullptr,
                    "PowerDissipationQoI: The pointer to the solver should be different than nullptr!");
        solver=solver_;

        nf=new mfem::ParNonlinearForm(solver->GetVelocityFES());
        intg=new PowerDissipationIntegrator(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
        nf->AddDomainIntegrator(intg);

        bntg=new BrinkmanPowerDissipation(solver->GetVelocity(),
                                          *(solver->GetTargetBrinkmanPenal()));
        lf=new ParNonlinearForm(solver->GetDesignFES());
        lf->AddDomainIntegrator(bntg);
    }

    ~PowerDissipationTGQoI()
    {
        delete nf;
        delete lf;
    }

    double Eval()
    {
        mfem::BlockVector& sol=solver->GetSol();
        intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
        double rez = nf->GetEnergy(sol.GetBlock(0)); //we need only the velocities
        return rez;
    }

    void Grad(mfem::Vector& grad)
    {
        tmpv.SetSize(grad.Size());
        //gradients with respect to velocity
        mfem::BlockVector arhs(solver->GetSol());
        arhs=0.0;
        intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
        //compute derivative of the objective
        nf->Mult(solver->GetSol().GetBlock(0),arhs.GetBlock(0));
        solver->ASolve(arhs);
        solver->GradD(tmpv);

        bntg->SetVelocity(solver->GetVelocity());
        bntg->SetBrinkmanCoefficient(*(solver->GetTargetBrinkmanPenal()));
        lf->Mult(tmpv,grad);
        grad.Add(1.0,tmpv);
    }

    double ModelError(mfem::GridFunction& moderr)
    {
        //gradients with respect to velocity
        mfem::BlockVector arhs(solver->GetSol());
        arhs=0.0;
        intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
        //compute derivative of the objective
        nf->Mult(solver->GetSol().GetBlock(0),arhs.GetBlock(0));
        //compute the adjoint
        solver->ASolve(arhs);
        return solver->ModelErrors(moderr);
    }

    double DiscretizationError(mfem::GridFunction& diserr)
    {
        //compute the adjoint
        {
            mfem::BlockVector arhs(solver->GetSol());
            arhs=0.0;
            intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
            nf->Mult(solver->GetSol().GetBlock(0),arhs.GetBlock(0));
            solver->ASolve(arhs);
        }

        //create a new Stokes solver
        mfem::StokesSolver* hsolver=new mfem::StokesSolver(solver->GetMesh(),
                                                           (solver->GetVelocityFES()->GetOrder(0))+1 );
        solver->TransferData(hsolver);
        //transfer the forward and the adjoint solutions
        {
            mfem::ParGridFunction& vel=hsolver->GetVelocity();
            vel.ProjectGridFunction(solver->GetVelocity());
            vel.GetTrueDofs(hsolver->GetSol().GetBlock(0));

            mfem::ParGridFunction& pre=hsolver->GetPressure();
            pre.ProjectGridFunction(solver->GetPressure());
            pre.GetTrueDofs(hsolver->GetSol().GetBlock(1));

            vel=hsolver->GetAVelocity();
            vel.ProjectGridFunction(solver->GetAVelocity());
            vel.GetTrueDofs(hsolver->GetAdj().GetBlock(0));

            pre=hsolver->GetAPressure();
            pre.ProjectGridFunction(solver->GetAPressure());
            pre.GetTrueDofs(hsolver->GetAdj().GetBlock(1));
        }

        MPI_Comm lcomm;
        lcomm=solver->GetMesh()->GetComm();
        int myrank;
        MPI_Comm_rank(lcomm,&myrank);

        if(myrank==0){std::cout<<"Solution HSolver"<<std::endl;}

        hsolver->FSolve(); //Improved forward solution
        //Improved adjoint solution
        PowerDissipationTGQoI* hqoi=new PowerDissipationTGQoI(hsolver);
        {
            mfem::BlockVector arhs(hsolver->GetSol());
            arhs=0.0;
            //the coefficients are the same for solver and hsolver
            hqoi->intg->SetParameters(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());
            hqoi->nf->Mult(hsolver->GetSol().GetBlock(0),arhs.GetBlock(0));
            hsolver->ASolve(arhs);
        }

        double hval=hqoi->Eval();
        if(myrank==0){std::cout<<"HQoi="<<hval<<std::endl;}

        delete hqoi;

        {
            mfem::ParLinearForm* lf=new mfem::ParLinearForm(solver->GetErrorFES());
            PowerDissipationIntegratorErr* lint=
                    new PowerDissipationIntegratorErr(solver->GetViscosity(),solver->GetTargetBrinkmanPenal());

            lint->SetGridFunctions(solver->GetVelocity(),hsolver->GetVelocity());

            StokesResidualIntegrator* rint=
                    new StokesResidualIntegrator(solver->GetViscosity(), solver->GetBrinkmanPenal(),
                                                 solver->GetVolForces());

            rint->SetLORFields(solver->GetVelocity(),solver->GetPressure(),
                               solver->GetAVelocity(), solver->GetAPressure());
            rint->SetHORFields(hsolver->GetVelocity(),hsolver->GetPressure(),
                               hsolver->GetAVelocity(),hsolver->GetAPressure());


            lf->AddDomainIntegrator(rint);
            lf->AddDomainIntegrator(lint);
            lf->Assemble();

            diserr.SetSpace(solver->GetErrorFES());
            diserr=0.0;

            lf->ParallelAssemble(diserr);
            delete lf;
        }

        double locres=0.0;
        for(int i=0;i<diserr.Size();i++)
        {
            locres=locres+diserr[i];
        }

        double totres=0.0;

        MPI_Allreduce(&locres, &totres, 1, MPI_DOUBLE,MPI_SUM,lcomm);

        delete hsolver;

        return totres;
    }


private:
    mfem::StokesSolver* solver;

    mfem::ParNonlinearForm* nf;
    mfem::PowerDissipationIntegrator* intg;

    mfem::BrinkmanPowerDissipation* bntg;
    mfem::ParNonlinearForm* lf;

    mfem::Vector tmpv;
};


class MassSolver:public mfem::Solver
{
private:
    MassSolver(mfem::ParFiniteElementSpace* fes_):fes(fes_),mcoeff(1.0)
    {
        bf=new mfem::ParBilinearForm(fes);
        bf->AddDomainIntegrator(new mfem::MassIntegrator(mcoeff));
        bf->SetAssemblyLevel(mfem::AssemblyLevel::FULL);
        bf->Assemble();
        M=bf->ParallelAssemble();
        prec=new mfem::HypreBoomerAMG(*M);
        pcg=new mfem::CGSolver(fes->GetComm());
        pcg->SetPreconditioner(*prec);
        SetSolver();
        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);
    }

    virtual
    ~MassSolver()
    {
        delete pcg;
        delete prec;
        delete bf;
    }

    virtual
    void Mult(const Vector &x, Vector &y) const
    {
        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);
        pcg->Mult(x,y);
    }

    void SetSolver(double rtol_=1e-8, double atol_=1e-12,int miter_=1000, int prt_level_=1)
    {
        rtol=rtol_;
        atol=atol_;
        max_iter=miter_;
        prt_level=prt_level_;
    }


    void Update()
    {
        delete pcg;
        delete prec;
        delete bf;
        fes->Update();
        bf=new mfem::ParBilinearForm(fes);
        bf->AddDomainIntegrator(new mfem::MassIntegrator(mcoeff));
        bf->SetAssemblyLevel(mfem::AssemblyLevel::FULL);
        bf->Assemble();
        M=bf->ParallelAssemble();
        prec=new mfem::HypreBoomerAMG(*M);
        pcg=new mfem::CGSolver(fes->GetComm());
        pcg->SetPreconditioner(*prec);
        pcg->SetAbsTol(atol);
        pcg->SetRelTol(rtol);
        pcg->SetMaxIter(max_iter);
        pcg->SetPrintLevel(prt_level);
    }

public:
    mfem::ParFiniteElementSpace *fes;
    mfem::ConstantCoefficient mcoeff;
    mfem::ParBilinearForm* bf;
    mfem::HypreParMatrix* M;
    mfem::Solver* prec;
    mfem::CGSolver* pcg;

    double atol;
    double rtol;
    int max_iter;
    int prt_level;


};



}


#endif
