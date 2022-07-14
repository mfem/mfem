#ifndef COEFFICIENTS_HPP
#define COEFFICIENTS_HPP


#include "mfem.hpp"


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
            st(1,2)=cc*(1.0-2.0*nu)*ss(1,2); st(2,1)=st(1,2);
            st(0,2)=cc*(1.0-2.0*nu)*ss(0,2); st(2,0)=st(0,2);
            st(0,1)=cc*(1.0-2.0*nu)*ss(0,1); st(1,0)=st(0,1);
        }else{
            st(0,0)=cc*((1.0-nu)*ss(0,0)+nu*ss(1,1));
            st(1,1)=cc*(nu*ss(0,0)+(1.0-nu)*ss(1,1));
            //st(2,2)=cc*(nu*ss(0,0)+nu*ss(1,1));
            st(0,1)=cc*(1.0-2.0*nu)*ss(0,1); st(1,0)=st(0,1);
        }
    }
private:
    double E;
    double nu;
};



};

#endif
