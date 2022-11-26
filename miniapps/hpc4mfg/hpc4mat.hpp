#ifndef HPC4MAT_HPP
#define HPC4MAT_HPP

#include "../autodiff/admfem.hpp"
#include "mfem.hpp"


namespace mfem{


class BasicNLDiffusionCoefficient{
public:


    virtual
    ~BasicNLDiffusionCoefficient(){}

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)=0;

    virtual
    void Grad(ElementTransformation &T,
              const IntegrationPoint &ip,
              const Vector& u, Vector& r)=0;

    virtual
    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)=0;

    virtual
    void GradWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, Vector& h)=0;

private:
};


class ExampleNLDiffusionCoefficient:public BasicNLDiffusionCoefficient
{
public:

    ExampleNLDiffusionCoefficient(double a_=1.0, double b_=1.0, double c_=1.0, double d_=0.1){
        ownership=true;
        a=new ConstantCoefficient(a_);
        b=new ConstantCoefficient(b_);
        c=new ConstantCoefficient(c_);
        d=new ConstantCoefficient(d_);
    }

    ExampleNLDiffusionCoefficient(Coefficient& a_, Coefficient& b_,
                                  Coefficient& c_, Coefficient& d_)
    {
        ownership=false;
        a=&a_;
        b=&b_;
        c=&c_;
        d=&d_;
    }

    virtual
    ~ExampleNLDiffusionCoefficient()
    {
        if(ownership)
        {
            delete a;
            delete b;
            delete c;
            delete d;
        }
    }

    virtual
    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);

        return 0.5*aa*(du*du)+0.5*bb*(u[dim]-cc)*(u[dim]-cc)+0.5*dd*exp(du*du);
    }

    virtual
    void Grad(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u, Vector& r)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);
        double nd=du*du;

        r.SetSize(dim+1); r=0.0;

        for(int i=0; i<dim; i++){
            r[i]=aa*du[i]+dd*exp(nd)*du[i];
        }
        r[dim]=bb*(u[dim]-cc);
    }

    virtual
    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        aa=a->Eval(T,ip);
        bb=b->Eval(T,ip);
        cc=c->Eval(T,ip);
        dd=d->Eval(T,ip);

        int dim=u.Size()-1;
        Vector du(u.GetData(),dim);
        double nd=du*du;

        h.SetSize(dim+1,dim+1); h=0.0;

        for(int i=0;i<dim;i++){
            h(i,i)=aa+dd*exp(nd);
            for(int j=0;j<dim;j++){
                h(i,j)=h(i,j)+2.0*dd*exp(nd)*du[i]*du[j];
            }
        }

    }

    void GradWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, Vector& h){};


private:
    bool ownership;
    Coefficient* a;
    Coefficient* b;
    Coefficient* c;
    Coefficient* d;

    double aa;
    double bb;
    double cc;
    double dd;

};

class SurrogateNLDiffusionCoefficient:public BasicNLDiffusionCoefficient
{
public:

    class MyNeuralNet
    {
    public:

        MyNeuralNet()
        {
           this->readSurrogateModel();
        }

        template<typename dtype>
        void Eval(dtype* inp, dtype* out)
        {
            dtype rr[A1.Height()];
            for(int i=0;i<A1.Height();i++)
            {
                rr[i]=(b1[i]);
                for(int j=0;j<A1.Width();j++){
                   rr[i]=rr[i]+A1(i,j)*inp[j];
                }
                rr[i]=pow(rr[i],2.0);
                rr[i]=exp(-rr[i]);          // activation function
            }

            for(int i=0;i<A2.Height();i++)
            {
                out[i]=(b2[i]);
                for(int j=0;j<A2.Width();j++){
                  out[i]=out[i]+A2(i,j)*rr[j];
                }
            }
            //A1.Print();
        }

        
        void Eval(mfem::Vector& inp, mfem::Vector& out)
        {
            Eval(inp.GetData(),out.GetData());
        }
        
        void EvalGrad(mfem::Vector& inp, mfem::Vector& out)
        {
            mfem::internal::dual<double,double> adinp[inp.Size()];
            mfem::internal::dual<double,double> adout[out.Size()];

            for(int i=0;i<inp.Size();i++)
            {
                adinp[i] = inp[i];
                adinp[i].gradient=0.0;
            }

            // threshold is last entry(inp.Size()-1). Set this one to 1.0
            adinp[inp.Size()-1].gradient=1.0;

            Eval(adinp,adout);

            for(int i=0;i<out.Size();i++)
            {
                out[i]=adout[i].gradient;
                //std::cout<<out[i]<<std::endl;
            }
        }

    private:
        mfem::DenseMatrix A1;
        mfem::DenseMatrix A2;
        mfem::Vector b1;
        mfem::Vector b2;

        int Weight1Rows;
        int Weight2Rows;

    void readSurrogateModel();
    };

    class MyNeuralNet_MatTensor
    {
    public:

        MyNeuralNet_MatTensor()
        {
           this->readSurrogateModel();
        }

        template<typename dtype>
        void Eval(dtype* inp, dtype* out)
        {
            dtype rr[A1.Height()];
            for(int i=0;i<A1.Height();i++)
            {
                 rr[i]=(b1[i]);
                 for(int j=0;j<A1.Width();j++){
                    rr[i]=rr[i]+A1(i,j)*inp[j];
                 }
                 rr[i]=pow(rr[i],2.0);
                 rr[i]=exp(-rr[i]);          // activation function
            }

            for(int i=0;i<A2.Height();i++)
            {
                out[i]=b2[i];
                 for(int j=0;j<A2.Width();j++){
                   out[i]=out[i]+A2(i,j)*rr[j];
                 }
             }
            //A1.Print();
        }


        void Eval(mfem::Vector& inp, mfem::Vector& out)
        {
            Eval<double>(inp.GetData(),out.GetData());
        }

        void EvalGrad(mfem::Vector& inp, mfem::Vector& out)
        {
            mfem::internal::dual<double,double> adinp[inp.Size()];
            mfem::internal::dual<double,double> adout[out.Size()];

            for(int i=0;i<inp.Size();i++)
            {
                adinp[i] = (inp[i]);
                adinp[i].gradient=0.0;
            }

            // threshold is last entry(inp.Size()-1). Set this one to 1.0
            adinp[inp.Size()-1].gradient=1.0;

            Eval(adinp,adout);

            for(int i=0;i<out.Size();i++)
            {
                out[i]=adout[i].gradient;
            }
        }

    private:
        mfem::DenseMatrix A1;
        mfem::DenseMatrix A2;
        mfem::Vector b1;
        mfem::Vector b2;

        int Weight1Rows;
        int Weight2Rows;

    void readSurrogateModel();

    };

    SurrogateNLDiffusionCoefficient( )
    {
             tNeurolNetTensor   = new MyNeuralNet_MatTensor;
             tNeurolNet         = new MyNeuralNet;
    };

    SurrogateNLDiffusionCoefficient( 
        std::string & GradName,
        std::string & HessianName )

    {            
        tNeurolNetTensor   = new MyNeuralNet_MatTensor;
        tNeurolNet         = new MyNeuralNet;

    };

    ~SurrogateNLDiffusionCoefficient()
    {
        delete tNeurolNetTensor;
        delete tNeurolNet;
    }

    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)
    {
        return 0.0;
    }

    void Grad(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u, Vector& r)
    {
        if(false)
        {
            int dim=u.Size();
            mfem::Vector du(dim);
            du=u;
            //mfem::Vector du(u.GetData(),dim);

           // mfem::Vector drr(dim); drr = 0.0;

             du[0] = du[0];
             du[1] = du[1];

            tNeurolNet->Eval( du, r);

            //drr.Print();
        }
        else
        {
            int dim=u.Size();

            int NumIG = 40;

            mfem::Vector dr1((dim-1)*(dim-1));

            r =0.0;

            for(int Ik = 1; Ik <= NumIG; Ik++)
            {
                mfem::Vector du(dim);
                du=u;

                mfem::Vector duu(dim);
                duu=u;
                duu[0] = u[0]/NumIG;
                duu[1] = u[1]/NumIG;
                
                dr1 = 0.0;
                du[0] = u[0]/NumIG*Ik;
                du[1] = u[1]/NumIG*Ik;

                tNeurolNetTensor->Eval( du, dr1);

                mfem::DenseMatrix LMat(dim-1);  LMat = 0.0;
                mfem::DenseMatrix SkewMat(dim-1);  SkewMat = 0.0;
                mfem::DenseMatrix LLT(dim-1);  LLT = 0.0;

                // tril
                LMat(0,0) = dr1( 0 );
                LMat(1,0) = dr1( 1 );
                LMat(1,1) = dr1( 2 );
  
                MultAAt( LMat, LLT );

                // triu
                SkewMat(0,1) = dr1( 3 );

                mfem::DenseMatrix SkewMatTrans;
                SkewMatTrans.Transpose(SkewMat);

                mfem::DenseMatrix h(dim); h = 0.0;
                LLT+=SkewMat;
                LLT -=SkewMatTrans;

                h(0,0) = LLT(0,0);
                h(1,0) = LLT(1,0);
                h(0,1) = LLT(0,1);
                h(1,1) = LLT(1,1);


                mfem::Vector vel(dim);
                h.Mult(duu,vel);

                //vel /= (double)NumIG;

                r[0] += vel[0];
                r[1] += vel[1];
            }
             //               r.Print();
            //    std::cout<<"-------"<<std::endl;

        }
    }

    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        int dim=u.Size()-1;
        mfem::Vector du(dim+1);
        du=u;

        mfem::Vector r(dim*dim);

         du[0] = du[0];
         du[1] = du[1];

        //mfem::mfem_error("check size");

        tNeurolNetTensor->Eval( du, r);

        if( 2 == dim )
        {
            mfem::DenseMatrix LMat(dim,dim);  LMat = 0.0;
            mfem::DenseMatrix SkewMat(dim,dim);  SkewMat = 0.0;
            mfem::DenseMatrix LLT(dim,dim);  LLT = 0.0;

            // tril
            LMat(0,0) = r( 0 );
            LMat(1,0) = r( 1 );
            LMat(1,1) = r( 2 );
  
            MultAAt( LMat, LLT );

            // triu
            SkewMat(0,1) = r( 3 );

            mfem::DenseMatrix SkewMatTrans;
            SkewMatTrans.Transpose(SkewMat);

            h = 0.0;
            LLT+=SkewMat;
            LLT -=SkewMatTrans;

            h(0,0) = LLT(0,0);
            h(1,0) = LLT(1,0);
            h(0,1) = LLT(0,1);
            h(1,1) = LLT(1,1);
        }
        else
        {
        // mfem::DenseMatrix LMat(dim,dim);  LMat = 0.0;
        // mfem::DenseMatrix SkewMat(dim,dim);  SkewMat = 0.0;
        // mfem::DenseMatrix LLT(dim,dim);  LLT = 0.0;

        // // tril
        // LMat(0,0) = OutputVec( 0 );
        // LMat(1,0) = OutputVec( 1 );
        // LMat(1,1) = OutputVec( 2 );
        // LMat(2,0) = OutputVec( 3 );
        // LMat(2,1) = OutputVec( 4 );
        // LMat(2,2) = OutputVec( 5 );

        // MultAAt( LMat, LLT );

        // // triu
        // SkewMat(0,1) = OutputVec( 6 );
        // SkewMat(0,2) = OutputVec( 7 );
        // SkewMat(1,2) = OutputVec( 8 );

        // mfem::DenseMatrix SkewMatTrans;
        // SkewMatTrans.Transpose(SkewMat);

        // h = 0.0;
        // h +=LLT;
        // h +=SkewMat;
        // h -=SkewMatTrans;
        }

    }

    void GradWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, Vector& h)
    {
        int dim=u.Size();
        mfem::Vector du(dim);
        du=u;

        du[0] = -du[0];
        du[1] = -du[1];

        tNeurolNet->EvalGrad( du, h);

        //h.Print();

    }


private:

    MyNeuralNet_MatTensor * tNeurolNetTensor = nullptr;
    MyNeuralNet           * tNeurolNet = nullptr;

};

class DarcyCoefficient:public BasicNLDiffusionCoefficient
{
public:

    DarcyCoefficient( )
    {
    };

    DarcyCoefficient( 
        std::string & GradName,
        std::string & HessianName )

    {            
    };

    ~DarcyCoefficient()
    {
    }

    double Eval(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u)
    {
        return 0.0;
    }

    void Grad(ElementTransformation &T,
                const IntegrationPoint &ip,
                const Vector& u, Vector& r)
    {
        int dim=u.Size()-1;

        if( isIsotorpic_ )
        {
            r[0] = -1.0 *(0.5001-u[dim]) / (DarcyScalar_) * u[0];
            r[1] = -1.0 *(0.5001-u[dim]) / (DarcyScalar_) * u[1];
        }
        else
        {
            r[0] = -2.0 *u[dim] / (DarcyScalar_) * u[0];
            r[1] = -1.0 *u[dim] / (DarcyScalar_) * u[1];
        }
    }

    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        int dim=u.Size()-1;

        if( 2 == dim )
        {
            if( isIsotorpic_ )
            {
                h(0,0) = -1.0*(0.5001-u[dim])/(DarcyScalar_);
                h(1,0) = 0.0;
                h(0,1) = 0.0;
                h(1,1) = -1.0*(0.5001-u[dim])/(DarcyScalar_);
            }
            else
            {
                h(0,0) = -2.0*u[dim]/(DarcyScalar_);
                h(1,0) = 0.0;
                h(0,1) = 0.0;
                h(1,1) = -1.0*u[dim]/(DarcyScalar_);
            }

        }
        else
        {

        }

    }

    void GradWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, Vector& h)
    {
        int dim=u.Size();

        if( isIsotorpic_ )
        {
            h[0] = 1.0 / (DarcyScalar_) * u[0];
            h[1] = 1.0 / (DarcyScalar_) * u[1];
        }
        else
        {
            h[0] = -2.0 / (DarcyScalar_) * u[0];
            h[1] = -1.0 / (DarcyScalar_) * u[1];
        }

    }


private:

    double DarcyScalar_ = 0.1;
    bool   isIsotorpic_ = true;
 
};

}

#endif

