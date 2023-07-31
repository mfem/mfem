#ifndef HPC4DIFFMAT_HPP
#define HPC4DIFFMAT_HPP

#include "../autodiff/admfem.hpp"
#include "mfem.hpp"


namespace mfem{


class BasicAdvDiffCoefficient{
public:

    virtual
    ~BasicAdvDiffCoefficient(){}

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
            const Vector& u, DenseMatrix& h)=0;

    virtual
    void GradWRTPreassureGrad(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, DenseMatrix& h, int SpatialDim) = 0;

private:
};

class SurrogateAdvDiffCoefficientnCoefficient : public BasicAdvDiffCoefficient
{
public:

    class MyNeuralNet
    {
    public:

        MyNeuralNet()
        {
           this->readSurrogateModel();
        }

        MyNeuralNet( std::string Name )
        {

            tStringWeight1 = "./NeuralNet/weights1_" + Name + ".txt";
            tStringWeight2 = "./NeuralNet/weights2_" + Name + ".txt";
            tStringBias1   = "./NeuralNet/bias1_" + Name + ".txt";
            tStringBias2   = "./NeuralNet/bias2_" + Name + ".txt";

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
                  //rr[i]=pow(rr[i],2.0);
                  //rr[i]=exp(-rr[i]);          // activation function
                //rr[i] = (exp(rr[i])-exp(-rr[i]))/(exp(-rr[i])+exp(rr[i]));
                rr[i] = -1.0 + 2.0/(1.0+exp(-2.0*rr[i]));

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

        template<typename dtype>
        void Eval(dtype* inp, double* inConst, dtype* out)
        {
            dtype rr[A1.Height()];
            for(int i=0;i<A1.Height();i++)
            {
                 rr[i]=(b1[i]);
                 for(int j=0;j<A1.Width();j++){
                    rr[i]=rr[i]+A1(i,j)*inp[j];
                 }
                  //rr[i]=pow(rr[i],2.0);
                  //rr[i]=exp(-rr[i]);          // activation function
                //rr[i] = (exp(rr[i])-exp(-rr[i]))/(exp(-rr[i])+exp(rr[i]));
                rr[i] = -1.0 + 2.0/(1.0+exp(-2.0*rr[i]));

            }

            dtype rr2[A2.Height()];
            for(int i=0;i<A2.Height();i++)
            {
                rr2[i]=b2[i];
                 for(int j=0;j<A2.Width();j++){
                   rr2[i]=rr2[i]+A2(i,j)*rr[j];
                 }
            }

                        mfem_error("SurrogateAdvDiffCoefficientnCoefficient::Grad() check impelemtation");

            out[0] = rr2[0] * rr2[0] * inConst[0] + rr2[0] * rr2[1] * inConst[1] + rr2[3]  * inConst[1];
            out[1] = rr2[0] * rr2[1] * inConst[0] - rr2[3]  * inConst[0] + rr2[1] * rr2[1] * inConst[1] +rr2[2] * rr2[2] * inConst[1];
            //A1.Print();
        }


        void Eval(mfem::Vector& inp, mfem::Vector& out)
        {
            Eval<double>(inp.GetData(),out.GetData());
        }

        void Eval(mfem::Vector& inp, mfem::Vector& inConst, mfem::Vector& out)
        {
            Eval<double>(inp.GetData(), inConst.GetData(), out.GetData());
        }

        void EvalGrad(mfem::Vector& inp, mfem::Vector& out )
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

        void EvalGrad(mfem::Vector& inp, mfem::Vector& out, int EntryInd )
        {
            mfem::internal::dual<double,double> adinp[inp.Size()];
            mfem::internal::dual<double,double> adout[out.Size()];

            for(int i=0;i<inp.Size();i++)
            {
                adinp[i] = (inp[i]);
                adinp[i].gradient=0.0;
            }

            adinp[EntryInd].gradient=1.0;

            Eval(adinp,adout);

            for(int i=0;i<out.Size();i++)
            {
                out[i]=adout[i].gradient;
            }
        }

        void EvalGrad(mfem::Vector& inp, mfem::Vector& inConst, mfem::Vector& out)
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

            Eval(adinp,inConst,adout);

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

        std::string tStringWeight1 = "./NeuralNet/weights1_AdvDiff.txt";
        std::string tStringWeight2 = "./NeuralNet/weights2_AdvDiff.txt";
        std::string tStringBias1   = "./NeuralNet/bias1_AdvDiff.txt";
        std::string tStringBias2   = "./NeuralNet/bias2_AdvDiff.txt";

    void readSurrogateModel();

    };

    SurrogateAdvDiffCoefficientnCoefficient( )
    {
             tNeurolNet   = new MyNeuralNet;
    };

    SurrogateAdvDiffCoefficientnCoefficient( 
        std::string & Name )

    { 
        tNeurolNet = new MyNeuralNet( Name );
    };

    ~SurrogateAdvDiffCoefficientnCoefficient()
    {
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
        this->Grad(u,r);
    }

    void Grad(const Vector& u, Vector& r)
    {
        mfem_error("SurrogateAdvDiffCoefficientnCoefficient::Grad() not implemented");
    }

    void Hessian(ElementTransformation &T,
                 const IntegrationPoint &ip,
                 const Vector& u, DenseMatrix& h)
    {
        this->Hessian(u,h);
    }

    void Hessian(const Vector& u, DenseMatrix& h)
    {
        int size=u.Size();
        int dim = 0;

        mfem::Vector tU = u;
        
        if( 3 == size )
        {
            dim = 2;
        }
        else{ mfem_error("SurrogateAdvDiffCoefficientnCoefficient::Hessian() wrong size"); }

        if( 2 == dim )
        {
            if(true)
            {
                mfem::Vector r(dim*dim);

                tNeurolNet->Eval( tU, r);

                h = 0.0;

                h(0,0) = r( 0 );
                h(0,1) = r( 1 );
                h(1,0) = r( 2 );
                h(1,1) = r( 3 );
            }
            else
            {
                mfem_error("SurrogateAdvDiffCoefficientnCoefficient::Hessian() ");
            }
        }
        else
        {
            mfem_error("SurrogateAdvDiffCoefficientnCoefficient::Hessian() 3D not implemented");
        }

    }

    void GradWRTDesing(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, DenseMatrix& h)
    {
        int size=u.Size();
        int dim = 0;

        mfem::Vector tU = u;
        
        if( 3 == size )
        {
            dim = 2;
        }
        else{ mfem_error("SurrogateAdvDiffCoefficientnCoefficient::GradWRTDesing() wrong size"); }

        if(false)
        {

        }
        else{
            //mfem_error("SurrogateAdvDiffCoefficientnCoefficient::GradWRTDesing()  not implemented");
               
                mfem::Vector r(dim*dim);

                tNeurolNet->EvalGrad( tU, r);

                h = 0.0;

                h(0,0) = r( 0 );
                h(0,1) = r( 1 );
                h(1,0) = r( 2 );
                h(1,1) = r( 3 );
            }
    }

    void GradWRTPreassureGrad(ElementTransformation &T,
            const IntegrationPoint &ip,
            const Vector& u, DenseMatrix& h, int SpatialDim)
    {
        int size=u.Size();
        int dim = 0;

        mfem::Vector tU = u;
        
        if( 3 == size )
        {
            dim = 2;
        }
        else{ mfem_error("SurrogateAdvDiffCoefficientnCoefficient::GradWRTDesing() wrong size"); }
               
        mfem::Vector r(dim*dim);

        tNeurolNet->EvalGrad( tU, r, SpatialDim);

        h = 0.0;

        h(0,0) = r( 0 );
        h(0,1) = r( 1 );
        h(1,0) = r( 2 );
        h(1,1) = r( 3 );
    }

private:

    MyNeuralNet           * tNeurolNet = nullptr;

};


}



#endif

