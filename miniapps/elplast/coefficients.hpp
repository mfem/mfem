#ifndef COEFFICIENTS_HPP
#define COEFFICIENTS_HPP


#include "mfem.hpp"
#include "linalg/dual.hpp"

namespace mfem {


class StrainCoefficient:public VectorCoefficient{
public:
    StrainCoefficient(GridFunction& disp_):VectorCoefficient(disp_.VectorDim()*(disp_.VectorDim()+1)/2)
    {
        disp=&disp_;
        g.SetSize(disp->VectorDim());
        e.SetSize(disp->VectorDim());
    }

    virtual
    ~StrainCoefficient(){}

    virtual
    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip)
    {
        T.SetIntPoint(&ip);
        disp->GetVectorGradient(T,g);
        for(int i=0;i<disp->VectorDim();i++){
        for(int j=i;j<disp->VectorDim();j++){
            e(i,j)=0.5*(g(i,j)+g(j,i));
        }}

        V.SetSize(GetVDim());

        if(disp->VectorDim()==3){
            V(0)=e(0,0);
            V(1)=e(1,1);
            V(2)=e(2,2);
            V(3)=e(1,2);
            V(4)=e(0,2);
            V(5)=e(0,1);
        }else{
            V(0)=e(0,0);
            V(1)=e(1,1);
            V(2)=e(0,1);
        }
    }

private:
    GridFunction* disp;
    DenseMatrix g;
    DenseMatrix e;

};


class EngStrainCoefficient: public mfem::VectorCoefficient{
public:

    EngStrainCoefficient(GridFunction& disp_):VectorCoefficient(disp_.VectorDim()*(disp_.VectorDim()+1)/2)
    {
        disp=&disp_;
        g.SetSize(disp->VectorDim());
        e.SetSize(disp->VectorDim());
    }

    virtual
    ~EngStrainCoefficient(){}

    virtual
    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip)
    {
        T.SetIntPoint(&ip);
        disp->GetVectorGradient(T,g);
        for(int i=0;i<disp->VectorDim();i++){
        for(int j=i;j<disp->VectorDim();j++){
            e(i,j)=0.5*(g(i,j)+g(j,i));
        }}

        V.SetSize(GetVDim());

        if(disp->VectorDim()==3){
            V(0)=e(0,0);
            V(1)=e(1,1);
            V(2)=e(2,2);
            V(3)=2.0*e(1,2);
            V(4)=2.0*e(0,2);
            V(5)=2.0*e(0,1);
        }else{
            V(0)=e(0,0);
            V(1)=e(1,1);
            V(2)=2.0*e(0,1);
        }
    }


private:
    GridFunction* disp;
    DenseMatrix g;
    DenseMatrix e;


};

template<typename ElastMaterial>
class StressCoefficient:public VectorCoefficient
{
public:
    StressCoefficient(ElastMaterial& mat_, MatrixCoefficient& str_):
        VectorCoefficient(str_.GetWidth()*(str_.GetWidth()+1)/2)
    {
        mat=&mat_;
        str=&str_;
        ss.SetSize(str_.GetWidth());

    }

    virtual
    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip)
    {
        str->Eval(ss,T,ip);
        mat->EvalStress(st,ss,T,ip);
        if(str->GetWidth()==3){
            V(0)=st(0,0);
            V(1)=st(1,1);
            V(2)=st(2,2);
            V(3)=st(1,2);
            V(4)=st(0,2);
            V(5)=st(0,1);
        }else{
            V(0)=st(0,0);
            V(1)=st(1,1);
            V(2)=st(0,1);
        }
    }

private:
    ElastMaterial* mat;
    MatrixCoefficient* str;

    DenseMatrix ss; //strain
    DenseMatrix st; //stress
};

template<typename ElastMaterial>
class VonMisesStressCoefficient:public Coefficient
{
public:
    VonMisesStressCoefficient(ElastMaterial& mat_, MatrixCoefficient& str_){
        mat=&mat_;
        str=&str_;//stress
        ss.SetSize(str->GetWidth());
        st.SetSize(str->GetWidth());
    }

    virtual
    double Eval(ElementTransformation &T,
                 const IntegrationPoint &ip)
    {
        str->Eval(ss,T,ip);
        mat->EvalStress(st,ss,T,ip);

        double res=0.0;

        if(str->GetWidth()==3){
            res=(st(0,0)-st(1,1))*(st(0,0)-st(1,1));
            res=res+(st(1,1)-st(2,2))*(st(1,1)-st(2,2));
            res=res+(st(0,0)-st(2,2))*(st(0,0)-st(2,2));
            res=res+6*st(1,2)*st(1,2);
            res=res+6*st(0,2)*st(0,2);
            res=res+6*st(0,1)*st(0,1);
        }else{
            res=(st(0,0)-st(1,1))*(st(0,0)-st(1,1));
            res=res+st(1,1)*st(1,1);
            res=res+st(0,0)*st(0,0);
            res=res+6*st(0,1)*st(0,1);
        }

        return sqrt(res);

    }

private:
   ElastMaterial* mat;
   MatrixCoefficient* str;
   DenseMatrix st; //stress
   DenseMatrix ss; //strain

};

class IsoElastMat
{
public:
    IsoElastMat()
    {
        E=1.0;
        nu=0.2;
    }

    void SetLameParam(double lam, double mu)
    {
        E=mu*(3.0*lam+2.0*mu)/(lam+mu);
        nu=lam/(2.0*(lam+mu));
    }

    void SetElastParam(double E_,double nu_)
    {
        E=E_;
        nu=nu_;
    }

    void SetE(double E_){ E=E_;}

    void SetPoisson(double nu_){ nu=nu_;}


    void EvalStress(DenseMatrix st, DenseMatrix ss,
                    ElementTransformation &T,
                    const IntegrationPoint &ip)
    {
        double cc=E/((1.0+nu)*(1.0-2.0*nu));
        if(ss.Width()==3){
            st(0,0)=cc*((1.0-nu)*ss(0,0)+nu*ss(1,1)+nu*ss(2,2));
            st(1,1)=cc*(nu*ss(0,0)+(1.0-nu)*ss(1,1)+nu*ss(2,2));
            st(2,2)=cc*(nu*ss(0,0)+nu*ss(1,1)+(1.0-nu)*ss(2,2));
            st(1,2)=cc*(1.0-2.0*nu)*ss(1,2); st(2,1)=cc*(1.0-2.0*nu)*ss(2,1);
            st(0,2)=cc*(1.0-2.0*nu)*ss(0,2); st(2,0)=cc*(1.0-2.0*nu)*ss(2,0);
            st(0,1)=cc*(1.0-2.0*nu)*ss(0,1); st(1,0)=cc*(1.0-2.0*nu)*ss(1,0);
        }else{
            st(0,0)=cc*((1.0-nu)*ss(0,0)+nu*ss(1,1));
            st(1,1)=cc*(nu*ss(0,0)+(1.0-nu)*ss(1,1));
            //st(2,2)=cc*(nu*ss(0,0)+nu*ss(1,1));
            st(0,1)=cc*(1.0-2.0*nu)*ss(0,1); st(1,0)=cc*(1.0-2.0*nu)*ss(1,0);
        }
    }

    template<typename dtype>
    void EvalStress(dtype* st, dtype* ss,
                    ElementTransformation &T,
                    const IntegrationPoint &ip)
    {
        double cc=E/((1.0+nu)*(1.0-2.0*nu));
        st[0+3*0]=cc*((1.0-nu)*ss[0+3*0]+nu*ss[1+3*1]+nu*ss[2+3*2]);
        st[1+3*1]=cc*(nu*ss[0+3*0]+(1.0-nu)*ss[1+3*1]+nu*ss[2+3*2]);
        st[2+3*2]=cc*(nu*ss[0+3*0]+nu*ss[1+3*1]+(1.0-nu)*ss[2+3*2]);
        st[1+3*2]=cc*(1.0-2.0*nu)*ss[1+3*2]; st[2+3*1]=cc*(1.0-2.0*nu)*ss[2+3*1];
        st[0+3*2]=cc*(1.0-2.0*nu)*ss[0+3*2]; st[2+3*0]=cc*(1.0-2.0*nu)*ss[2+3*0];
        st[0+3*1]=cc*(1.0-2.0*nu)*ss[0+3*1]; st[1+3*0]=cc*(1.0-2.0*nu)*ss[1+3*0];

    }

    // mat[9x9] ss[9]
    template<typename dtype>
    void EvalGrad(dtype* mat, dtype* ss,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
    {
        double cc=E/((1.0+nu)*(1.0-2.0*nu));
        for(int i=0;i<81;i++){ mat[i]=dtype(0.0);}

        mat[0*9 + 0]=cc*(1.0-nu);
        mat[0*9 + 4]=cc*nu;
        mat[0*9 + 8]=cc*nu;

        mat[4*9 + 0]=cc*nu;
        mat[4*9 + 4]=cc*(1.0-nu);
        mat[4*9 + 8]=cc*nu;

        mat[8*9+  0]=cc*nu;
        mat[8*9+  4]=cc*nu;
        mat[8*9+  8]=cc*(1.0-nu);

        mat[1+3*2+(1+3*2)*9]=cc*(1.0-2.0*nu);
        mat[0+3*2+(0+3*2)*9]=cc*(1.0-2.0*nu);
        mat[0+3*1+(0+3*1)*9]=cc*(1.0-2.0*nu);

        mat[2+3*1+(2+3*1)*9]=cc*(1.0-2.0*nu);
        mat[2+3*0+(2+3*0)*9]=cc*(1.0-2.0*nu);
        mat[1+3*0+(1+3*0)*9]=cc*(1.0-2.0*nu);
    }


    void EvalGrad(DenseMatrix& mat,
                  Vector& ee, // strains
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
    {
        mat.SetSize(9);
        EvalGrad(mat.GetData(), ee.GetData(), T, ip);
    }

private:
    double E;
    double nu;
};


class J2YieldFunction
{
public:

    J2YieldFunction()
    {
        sigma_0=1.0;
        H=0.0;
        beta=0.0;
    }

    J2YieldFunction(double ssy_,double H_, double beta_)
    {
        sigma_0=ssy_;
        H=H_;
        beta=beta_;
    }

    void Set(double ssy_,double H_, double beta_)
    {
        sigma_0=ssy_;
        H=H_;
        beta=beta_;
    }

    void SetJ2Param(double ssy_, double H_, double beta_)
    {
        sigma_0=ssy_;
        H=H_;
        beta=beta_;
    }

    //str[9] - stress tensor
    //ip[2]:
    //      ip[0] accumulated plastic strain (possitive)
    //      ip[1] filtered accumulated plastic strain (possitive)
    template<typename dtype0,typename dtype1,typename dtype2 >
    dtype0 EvalI(dtype1* str,dtype2* ip)
    {
        dtype0 p=(str[0]+str[4]+str[8])/3.0;
        dtype0 dstr[9];
        for(int i=0;i<9;i++){ dstr[i]=str[i];}
        dstr[0]=dstr[0]-p;
        dstr[4]=dstr[4]-p;
        dstr[8]=dstr[8]-p;
        dtype0 se=dstr[0]*dstr[0];
        for(int i=1;i<9;i++){ se=se+dstr[i]*dstr[i];}
        se=sqrt(3.0*se/2.0);

        dtype0 sy=sigma_0+H*ip[0];
        dtype0 tv=exp(-beta*ip[1]);
        //return se-tv*tv*sy*sy;
        return se-tv*sy;
    }

    //return gradients of the yield function with respect to str and ip
    template<typename dtype>
    dtype EvalG(dtype* gstr, dtype* gip, dtype* str, dtype* ip)
    {
        dtype t1,t7,t9,t11,t13,t17,t19,t21,t23,t25,t26,t27,t29,t30,t40;
        t1 = str[0]*str[0];
        t7 = str[1]*str[1];
        t9 = str[2]*str[2];
        t11 = str[3]*str[3];
        t13 = str[4]*str[4];
        t17 = str[5]*str[5];
        t19 = str[6]*str[6];
        t21 = str[7]*str[7];
        t23 = str[8]*str[8];
        t25 = -4.0*str[0]*str[4]-4.0*str[0]*str[8]-4.0*str[4]*str[8]+4.0*t1+6.0*
                t11+4.0*t13+6.0*t17+6.0*t19+6.0*t21+4.0*t23+6.0*t7+6.0*t9;
        t26 = sqrt(t25);
        t27 = 1.0/t26;
        t29 = 4.0*str[4];
        t30 = 4.0*str[8];
        t40 = 4.0*str[0];
        gstr[0] = t27*(8.0*str[0]-t29-t30)/4.0;
        gstr[1] = 3.0*t27*str[1];
        gstr[2] = 3.0*t27*str[2];
        gstr[3] = 3.0*t27*str[3];
        gstr[4] = t27*(-t40+8.0*str[4]-t30)/4.0;
        gstr[5] = 3.0*t27*str[5];
        gstr[6] = 3.0*t27*str[6];
        gstr[7] = 3.0*t27*str[7];
        gstr[8] = t27*(-t40-t29+8.0*str[8])/4.0;

        gip[0] = -exp(-beta*ip[1])*H;
        gip[1] = beta*exp(-beta*ip[1])*(H*ip[0]+sigma_0);

        dtype sy=sigma_0+H*ip[0];
        dtype tv=exp(-beta*ip[1]);
        return t26/2.0-tv*sy;
    }

    double Eval(const Vector& str,const Vector& ivar)
    {
        return EvalI<double,double,double>(str.GetData(),ivar.GetData());
    }

    double Eval(double* str, double* ivar)
    {
        return EvalI<double,double,double>(str,ivar);
    }

    double EvalFGrad(Vector& drstr, Vector& divar,
                    Vector& str, Vector& ivar)
    {
        //drstr.SetSize(str.Size());
        //divar.SetSize(ivar.Size());
        typedef internal::dual<double, double> ADFloatType;
        ADFloatType tst[str.Size()];
        ADFloatType var[ivar.Size()];

        for(int i=0;i<str.Size();i++){
            tst[i].value=str[i];
            tst[i].gradient=0.0;
        }

        for(int i=0;i<ivar.Size();i++){
            var[i].value=ivar[i];
            var[i].gradient=0.0;
        }

        ADFloatType rez;
        for(int i=0;i<str.Size();i++){
            tst[i].gradient=1.0;
            rez=EvalI<ADFloatType,ADFloatType,ADFloatType>(tst,var);
            drstr[i]=rez.gradient;
            tst[i].gradient=0.0;
        }

        for(int i=0;i<ivar.Size();i++){
            var[i].gradient=1.0;
            rez=EvalI<ADFloatType,ADFloatType,ADFloatType>(tst,var);
            divar[i]=rez.gradient;
            var[i].gradient=0.0;
        }

        return rez.value;
    }

private:
    double H;
    double beta;
    double sigma_0;

};


class DruckerPragerYieldFunction
{
public:
    DruckerPragerYieldFunction(double ssy_,double alpha_)
    {
        sigma_0=ssy_;
        alpha=alpha_;
    }

    template<typename dtype0, typename dtype1, typename dtype2>
    dtype0 Eval(dtype1* str,dtype2* ip)
    {
        dtype0 p=(str[0]+str[4]+str[8])/3.0;
        dtype0 dstr[9];
        for(int i=0;i<9;i++){ dstr[i]=str[i];}
        dstr[0]=dstr[0]-p;
        dstr[4]=dstr[1]-p;
        dstr[8]=dstr[8]-p;
        dtype0 J2=dstr[0]*dstr[0];
        for(int i=1;i<9;i++){ J2=J2+dstr[i]*dstr[i];}
        J2=J2/2.0;
        return sqrt(J2)-3.0*alpha*p-sigma_0/sqrt(3.0);
    }

    double operator()(Vector& str,Vector& ivar)
    {
        return Eval<double>(str.GetData(),ivar.GetData());
    }

private:

    double sigma_0;
    double alpha;
};

class MatsuokaNakaiYieldFunction
{
public:
    MatsuokaNakaiYieldFunction(double phi_)
    {
        phi=phi_;
    }

    template<typename dtype0, typename dtype1, typename dtype2>
    dtype0 Eval(dtype1* str,dtype2* ip)
    {
        dtype0 I1=(str[0]+str[4]+str[8]);
        dtype0 I2=str[0]*str[4]+str[4]*str[8]+str[8]*str[0]
                              -str[3]*str[3]-str[6]*str[6]-str[7]*str[7];
        dtype0 I3= str[0]*str[4]*str[8]+str[1]*str[5]*str[6]+str[3]*str[7]*str[2]
                 -str[2]*str[4]*str[6]-str[1]*str[3]*str[8]-str[5]*str[7]*str[0];

        double tp=tan(phi);
        return I1*I2 - (9.0+8.0*tp*tp)*I3;
    }

    double operator()(Vector& str,Vector& ivar)
    {
        return Eval<double>(str.GetData(),ivar.GetData());
    }
private:
    double phi;
};





template<typename elmat, typename yfunc>
class StressEval
{
public:

    StressEval(elmat* mat_, yfunc* yf_, double lerr_=1e-8){
        mat=mat_;
        yf=yf_;
        lerr=lerr_;
    }

    void SetPlasticStrain(const Vector& ep_)
    {
        for(int i=0;i<9;i++){
            epn[i]=ep_[i];
        }
    }

    void SetInternalParameters(const Vector& ip_)
    {
        ipn[0]=ip_[0];
        ipn[1]=ip_[1];
    }

    void SetStrain(const Vector &ee_)
    {
        for(int i=0;i<9;i++){
            ee[i]=ee_[i];
        }
    }

    void EvalTangent(DenseMatrix& Cep,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
    {
        Vector ees(9);
        Vector css(9);
        for(int i=0;i<9;i++){
            ees[i]=ee[i]-epn[i];
        }

        mat->EvalStress(css.GetData(),ees.GetData(),T,ip);
        mat->EvalGrad(Cep,ees,T,ip);
        double f=yf->Eval(css.GetData(),ipn);
        if(f<0.0){
            return;}
        Vector dip(2);
        //elasto-plastic behaviour - modify the elastic tensor
        Vector r(9);
        //yf->EvalFGrad(r,dip,css,ipn);
        yf->EvalG(r.GetData(), dip.GetData(), css.GetData(),ipn);
        double H;

        mat->EvalStress(css.GetData(),r.GetData(),T,ip);
        H=(css*r)-dip[0]*sqrt(2.0*(r*r)/3.0);
        for(int i=0;i<9;i++){
        for(int j=0;j<9;j++){
            Cep(i,j)=Cep(i,j)-css(i)*css(j)/H;
        }}

    }

    /// vin[stress[9], strainp[9], ip[2], multiplier[1]]
    template<typename dtype>
    void EvalResidual(dtype* rr, dtype* vin,
                      ElementTransformation &T,
                      const IntegrationPoint &ip)
    {
        dtype* css=vin;   //current stress
        dtype* cep=vin+9; //current plastic strain
        dtype* cip=vin+18;//current internal variables
        dtype* lam=vin+20;//current Largrange multiplier

        dtype* rr1=rr;
        dtype* rr2=rr+9;
        dtype* rr3=rr+18;
        dtype* rr4=rr+20;

        dtype tv[9];
        dtype hh[2];
        dtype vv;

        //stress residual
        for(int i=0;i<9;i++){
            tv[i]=ee[i]-cep[i];
        }
        mat->EvalStress(rr1,tv,T,ip);
        for(int i=0;i<9;i++){
            rr1[i]=css[i]-rr1[i];
        }

        rr4[0]=yf->EvalG(tv, hh, css, cip);
        for(int i=0;i<9;i++){
            rr2[i]=cep[i]-epn[i]-(*lam)*tv[i];
        }

        //internal parameters
        vv=tv[0]*tv[0];
        for(int i=1;i<9;i++){ vv=vv+tv[i]*tv[i];}
        rr3[0]=cip[0]-ipn[0]-(*lam)*sqrt((2.0/3.0)*vv);
        rr3[1]=cip[1]-ipn[1];
    }

    //E
    void EvalNewton(DenseMatrix& tmat, Vector& rr,
                    Vector& vv,
                    ElementTransformation &T,
                    const IntegrationPoint &ip){
        typedef internal::dual<double, double> ADFloatType;
        ADFloatType tv[21];
        ADFloatType cv[21];
        for(int i=0;i<21;i++){
            cv[i].value=vv[i];
            cv[i].gradient=0.0;
        }

        for(int i=0;i<21;i++){
            cv[i].gradient=1.0;
            EvalResidual(tv, cv,T,ip);
            for(int j=0;j<21;j++){
                tmat(j,i)=tv[j].gradient;
            }
            cv[i].gradient=0.0;
        }

        for(int i=0;i<21;i++){
            rr[i]=tv[i].value;
        }
    }

    void Solve(Vector& css, Vector& cep, Vector& cip,
               ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        Vector ees(9);
        for(int i=0;i<9;i++){
            ees[i]=ee[i]-epn[i];
            cep[i]=epn[i];
        }

        cip[0]=ipn[0];
        cip[1]=ipn[1];

        mat->EvalStress(css.GetData(),ees.GetData(),T,ip);

        double f=yf->Eval(css,cip);
        if(f<0.0){return;}
        SolveEP(css,cep,cip,T,ip);
    }

    void SolveEP(Vector& css, Vector& cep, Vector& cip,
               ElementTransformation &T,
               const IntegrationPoint &ip, double err=1e-8){
        Vector vv(21);
        Vector rr(21);
        Vector tv(21);
        DenseMatrix mm(21);
        for(int i=0;i<9;i++){
            vv[i]=css[i];
            vv[i+9]=cep[i];
        }
        vv[18]=cip[0];
        vv[19]=cip[1];
        vv[20]=0.0;

        EvalNewton(mm,rr,vv,T,ip);

        /*
        std::cout<<std::endl;
        mm.PrintMatlab(std::cout);
        std::cout<<"rr="<<std::endl;
        rr.Print(std::cout,21);
        */

        double cerr=rr.Norml2();
        int it=0;
        while(cerr>err){
            DenseMatrixInverse im(mm);
            im.Mult(rr,tv);
            vv.Add(-1.0,tv);
            EvalNewton(mm,rr,vv,T,ip);
            cerr=rr.Norml2();
            //std::cout<<"it:"<<it<<" err="<<cerr<<std::endl;
            it++;
            if(it>100){
                mfem_error("Maximum number (100) of iterations have been"
                           "reached in the elasto-plastic material update! \n");
                break;}
        }

        for(int i=0;i<9;i++){
            css[i]=vv[i];
            cep[i]=vv[i+9];
        }
        cip[0]=vv[18];
        cip[1]=vv[19];
    }



private:
    elmat* mat;
    yfunc* yf;

    double ee[9]; //total strain
    double epn[9]; //plastic strain
    double ipn[2]; //internal variables

    double lerr;

};


}

#endif
