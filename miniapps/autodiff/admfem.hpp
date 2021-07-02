#ifndef ADMFEM_HPP 
#define ADMFEM_HPP

#include "mfem.hpp"
#include "fdual.hpp"
#include "tadvector.hpp"
#include "taddensemat.hpp"

#ifdef MFEM_USE_CODIPACK
#include <codi.hpp>

namespace mfem {

namespace ad {
#ifdef MFEM_USE_ADFORWARD
typedef codi::RealForward ADFloatType;
typedef TADVector<ADFloatType> ADVectorType;
typedef TADDenseMatrix<ADFloatType> ADMatrixType;
#else
typedef codi::RealReverse ADFloatType;
typedef TADVector<ADFloatType> ADVectorType;
typedef TADDenseMatrix<ADFloatType> ADMatrixType;
#endif
}


template<int vector_size=1, int state_size=1, int param_size=0>
class VectorFuncAutoDiff
{
public:
    VectorFuncAutoDiff(std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F_)
    {
            F=F_;
    }

    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }
            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                F(vparam,aduu,rr);
                for(int jj=0;jj<vector_size;jj++)
                {
                    jac(jj,ii)=rr[jj].getGradient();
                }
                aduu[ii].setGradient(0.0);
            }
        }
#else // use reverse mode
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ad::ADFloatType::TapeType& tape =ad::ADFloatType::getGlobalTape();
            typename ad::ADFloatType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }
            F(vparam,aduu,rr);
            for(int ii=0;ii<vector_size;ii++){ tape.registerOutput(rr[ii]); }
            tape.setPassive();

            for(int jj=0;jj<vector_size;jj++){
                rr[jj].setGradient(1.0);
                tape.evaluate();
                for(int ii=0;ii<state_size;ii++){
                    jac(jj,ii)=aduu[ii].getGradient();
                }
                tape.clearAdjoints();
                rr[jj].setGradient(0.0);
            }
            tape.reset(pos);
        }
#endif
    }
private:
    std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F;
}; //VectorFuncAutoDiff


template<template<typename, typename, typename, int, int, int> class CTD
         , int vector_size=1, int state_size=1, int param_size=0>
class QVectorFuncAutoDiff
{
public:
    /// Evaluates the vector function for given set of parameters and state
    /// values in vector uu. The result is returned in vector rr.
    void QVectorFunc(const mfem::Vector &vparam, mfem::Vector &uu, mfem::Vector& rr)
    {
        CTD<double,const mfem::Vector, mfem::Vector,
                        vector_size, state_size, param_size> tf;
        tf(vparam,uu,rr);
    }

    /// Returns the gradient of CTD(...) in the dense matrix jac.
    /// The dimensions of jac are vector_size x state_size, where state_size is the
    /// length of vector uu.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward mode

        CTD<ad::ADFloatType, const Vector, ad::ADVectorType,
                        vector_size, state_size, param_size> tf;
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }

            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                tf(vparam,aduu,rr);
                for(int jj=0;jj<vector_size;jj++)
                {
                    jac(jj,ii)=rr[jj].getGradient();
                }
                aduu[ii].setGradient(0.0);
            }
        }
#else //end MFEM_USE_ADFORWARD
        // use reverse mode
        CTD<ad::ADFloatType, const Vector, ad::ADVectorType,
                vector_size, state_size, param_size> tf;

        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
            ad::ADVectorType aduu(state_size);
            ad::ADVectorType rr(vector_size);
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ad::ADFloatType::TapeType& tape =ad::ADFloatType::getGlobalTape();
            typename ad::ADFloatType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }
            tf(vparam,aduu,rr);
            for(int ii=0;ii<vector_size;ii++){ tape.registerOutput(rr[ii]); }
            tape.setPassive();
            for(int jj=0;jj<vector_size;jj++){
                rr[jj].setGradient(1.0);
                tape.evaluate();
                for(int ii=0;ii<state_size;ii++){
                    jac(jj,ii)=aduu[ii].getGradient();
                }
                tape.clearAdjoints();
                rr[jj].setGradient(0.0);
            }
            tape.reset(pos);
        }
#endif
    }
};

template<template<typename, typename, typename, int, int> class CTD
         , int state_size=1, int param_size=0>
class QFunctionAutoDiff
{
public:

    /// Evaluates a function for arguments vparam and uu.
    /// The evaluatin is based on the operator() in the
    /// user provided functor CTD.
    double QEval(const Vector &vparam, Vector &uu)
    {
        CTD<double, const Vector, Vector, state_size, param_size> tf;
        return tf(vparam,uu);
    }

    /// Provides the same functionality as QGrad.
    void QVectorFunc(const Vector &vparam, Vector &uu, Vector &rr)
    {
        QGrad(vparam,uu,rr);
    }

    /// Returns the first derivative of CTD(...) with
    /// respect to the active arguments proved in vector uu.
    /// The length of rr is the same as for uu.
    void QGrad(const Vector &vparam, Vector &uu, Vector &rr)
    {

#ifdef MFEM_USE_ADFORWARD
        // use forward mode
        CTD<ad::ADFloatType, const Vector, ad::ADVectorType,
                        state_size, param_size> tf;
        rr.SetSize(state_size);
        {
            ad::ADVectorType aduu(state_size);
            for(int i=0;i<state_size;i++){
                aduu[i].setValue(uu[i]);
                aduu[i].setGradient(0.0);
            }

            ad::ADFloatType rez;

            for(int ii=0;ii<state_size;ii++){
                aduu[ii].setGradient(1.0);
                rez=tf(vparam,aduu);
                rr[ii]=rez.getGradient();
                aduu[ii].setGradient(0.0);
            }
        }

#else
        typedef codi::RealReverse ADFType;
        typedef TADVector<ADFType> ADFVector;

        CTD<ADFType, const Vector, ADFVector, state_size, param_size> tf;
        {
            ADFVector aduu(state_size);
            ADFType rez;
            for(int i=0;i<state_size;i++){
                aduu[i]=uu[i];
            }

            ADFType::TapeType& tape =ADFType::getGlobalTape();
            typename ADFType::TapeType::Position pos=tape.getPosition();

            tape.setActive();
            for(int ii=0;ii<state_size;ii++){ tape.registerInput(aduu[ii]); }

            rez=tf(vparam,aduu);
            tape.registerOutput(rez);
            tape.setPassive();

            rez.setGradient(1.0);
            tape.evaluate();
            for(int i=0;i<state_size;i++){
                rr[i]=aduu[i].getGradient();
            }
            tape.reset(pos);
        }
#endif
    }

    /// Provides same functionality as QHessian.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        QHessian(vparam,uu,jac);
    }

    /// Returns the Hessian of CTD(...) in the dense matrix hh.
    /// The dimensions of jac are state_size x state_size, where state_size is the
    /// length of vector uu.
    void QHessian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
#ifdef MFEM_USE_ADFORWARD
        // use forward-forward mode
        typedef codi::RealForwardGen<double>    ADFType;
        typedef TADVector<ADFType>              ADFVector;
        typedef TADDenseMatrix<ADFType>         ADFDenseMatrix;

        typedef codi::RealForwardGen<ADFType>   ADSType;
        typedef TADVector<ADSType>              ADSVector;
        typedef TADDenseMatrix<ADSType>         ADSDenseMatrix;

        CTD<ADSType, const Vector, ADSVector, state_size, param_size> tf;

        jac.SetSize(state_size);
        jac=0.0;
        {
            ADSVector aduu(state_size);
            for(int ii = 0;  ii < state_size; ii++)
            {
                aduu[ii].value().value()=uu[ii];
                aduu[ii].value().gradient()=0.0;
                aduu[ii].gradient().value()=0.0;
                aduu[ii].gradient().gradient()=0.0;
            }

            for(int ii = 0; ii < state_size; ii++)
            {
                aduu[ii].value().gradient()=1.0;
                for(int jj=0; jj<(ii+1); jj++)
                {
                    aduu[jj].gradient().value()=1.0;
                    ADSType rez=tf(vparam,aduu);
                    jac(ii,jj)=rez.gradient().gradient();
                    jac(jj,ii)=jac(ii,jj);
                    aduu[jj].gradient().value()=0.0;
                }
                aduu[ii].value().gradient()=0.0;
            }
        }
#else
        //use mixed forward and reverse mode
        typedef codi::RealForwardGen<double> ADFType;
        typedef TADVector<ADFType>           ADFVector;
        typedef TADDenseMatrix<ADFType>      ADFDenseMatrix;

        typedef codi::RealReverseGen<ADFType>   ADSType;
        typedef TADVector<ADSType>              ADSVector;
        typedef TADDenseMatrix<ADSType>         ADSDenseMatrix;

        CTD<ADSType, const Vector, ADSVector, state_size, param_size> tf;

        jac.SetSize(state_size);
        jac=0.0;
        {
            ADSVector aduu(state_size);
            for(int ii=0;ii < state_size ; ii++)
            {
                aduu[ii].value().value()=uu[ii];
            }

            ADSType rez;

            ADSType::TapeType& tape = ADSType::getGlobalTape();
            typename ADSType::TapeType::Position pos;
            for(int ii = 0; ii < state_size ; ii++)
            {
                pos=tape.getPosition();
                tape.setActive();

                for(int jj=0;jj < state_size; jj++) {
                    if(jj==ii) {aduu[jj].value().gradient()=1.0;}
                    else {aduu[jj].value().gradient()=0.0;}
                    tape.registerInput(aduu[jj]);
                }

                rez=tf(vparam,aduu);
                tape.registerOutput(rez);
                tape.setPassive();

                rez.gradient().value()=1.0;
                tape.evaluate();

                for(int jj=0; jj<(ii+1); jj++)
                {
                    jac(ii,jj)=aduu[jj].gradient().gradient();
                    jac(jj,ii)=jac(ii,jj);
                }
                tape.reset(pos);
            }

        }
#endif
    }
};

}
#else //end MFEM_USE_CODIPACK

//USE NATIVE IMPLEMENTATION
namespace mfem {

namespace ad {
typedef FDual<double> ADFloatType;
typedef TADVector<ADFloatType> ADVectorType;
typedef TADDenseMatrix<ADFloatType> ADMatrixType;
}

template<int vector_size=1, int state_size=1, int param_size=0>
class VectorFuncAutoDiff
{
public:
    VectorFuncAutoDiff(std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F_)
    {
            F=F_;
    }

public:
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
           ad::ADVectorType aduu(uu); //all dual numbers are initialized to zero
           ad::ADVectorType rr(vector_size);

           for (int ii = 0; ii < state_size; ii++)
           {
              aduu[ii].dual(1.0);
              F(vparam,aduu,rr);
              for (int jj = 0; jj < vector_size; jj++)
              {
                 jac(jj, ii) = rr[jj].dual();
              }
              aduu[ii].dual(0.0);
           }
        }
    }

private:
    std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)> F;

};


template<template<typename, typename, typename, int, int, int> class CTD
         , int vector_size=1, int state_size=1, int param_size=0>
class QVectorFuncAutoDiff
{
private:
    /// MFEM native forward AD-type
    typedef ad::FDual<double> ADFType;
    typedef TADVector<ADFType> ADFVector;
    typedef TADDenseMatrix<ADFType> ADFDenseMatrix;

public:
    /// Returns a vector valued function rr for supplied passive arguments
    /// vparam and active arguments uu. The evaluation is based on the
    /// user supplied CTD template class.
    void QVectorFunc(const Vector &vparam, Vector &uu, Vector &rr)
    {
       CTD<double, const Vector, Vector,
               vector_size, state_size, param_size> func;
       func(vparam, uu, rr);
    }

    /// Returns the gradient of CTD(...) residual in the dense matrix jac.
    /// The dimensions of jac are vector_size x state_size, where state_size is the
    /// length of vector uu.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        //use native AD package
        jac.SetSize(vector_size, state_size);
        jac = 0.0;
        {
           ADFVector aduu(uu); //all dual numbers are initialized to zero
           ADFVector rr(vector_size);

           for (int ii = 0; ii < state_size; ii++)
           {
              aduu[ii].dual(1.0);
              QEval(vparam, aduu, rr);
              for (int jj = 0; jj < vector_size; jj++)
              {
                 jac(jj, ii) = rr[jj].dual();
              }
              aduu[ii].dual(0.0);
           }
        }
    }

private:
    /// Evaluates the residual from CTD(...).
    /// Intended for internal use only.
    void QEval(const Vector &vparam, ADFVector &uu, ADFVector &rr)
    {
       CTD<ADFType, const Vector, ADFVector,
               vector_size, state_size, param_size> tf;
       tf(vparam, uu, rr);
    }
};

template<template<typename, typename, typename, int, int> class CTD
         , int state_size=1, int param_size=0>
class QFunctionAutoDiff
{
private:
    ///MFEM native AD-type for first derivatives
    typedef ad::FDual<double> ADFType;
    typedef TADVector<ADFType> ADFVector;
    typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
    ///MFEM native AD-type for second derivatives
    typedef ad::FDual<ADFType> ADSType;
    typedef TADVector<ADSType> ADSVector;
    typedef TADDenseMatrix<ADSType> ADSDenseMatrix;

public:
    /// Evaluates a function for arguments vparam and uu.
    /// The evaluatin is based on the operator() in the
    /// user provided functor CTD.
    double QEval(const Vector &vparam, Vector &uu)
    {
        CTD<double, const Vector, Vector, state_size, param_size> tf;
        return tf(vparam,uu);
    }

    /// Provides the same functionality as QGrad.
    void QVectorFunc(const Vector &vparam, Vector &uu, Vector &rr)
    {
        QGrad(vparam,uu,rr);
    }

    /// Returns the first derivative of CTD(...) with
    /// respect to the active arguments proved in vector uu.
    /// The length of rr is the same as for uu.
    void QGrad(const Vector &vparam, Vector &uu, Vector &rr)
    {
        int n = uu.Size();
        rr.SetSize(n);
        ADFVector aduu(uu);
        ADFType rez;
        for (int ii = 0; ii < n; ii++)
        {
           aduu[ii].dual(1.0);
           rez = QEval(vparam, aduu);
           rr[ii] = rez.dual();
           aduu[ii].dual(0.0);
        }
    }

    /// Provides same functionality as QHessian.
    void QJacobian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        QHessian(vparam,uu,jac);
    }

    /// Returns the Hessian of CTD(...) in the dense matrix hh.
    /// The dimensions of jac are state_size x state_size, where state_size is the
    /// length of vector uu.
    void QHessian(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        int n = uu.Size();
        jac.SetSize(n);
        jac = 0.0;
        {
           ADSVector aduu(n);
           for (int ii = 0; ii < n; ii++)
           {
              aduu[ii].real(ADFType(uu[ii], 0.0));
              aduu[ii].dual(ADFType(0.0, 0.0));
           }

           for (int ii = 0; ii < n; ii++)
           {
              aduu[ii].real(ADFType(uu[ii], 1.0));
              for (int jj = 0; jj < (ii + 1); jj++)
              {
                 aduu[jj].dual(ADFType(1.0, 0.0));
                 ADSType rez = QEval(vparam, aduu);
                 jac(ii, jj) = rez.dual().dual();
                 jac(jj, ii) = rez.dual().dual();
                 aduu[jj].dual(ADFType(0.0, 0.0));
              }
              aduu[ii].real(ADFType(uu[ii], 0.0));
           }
        }
    }

private:
    /// Evaluates the first derivative of CTD(...).
    /// Intended for internal use only.
    ADFType QEval(const Vector &vparam, ADFVector &uu)
    {
       CTD<ADFType, const Vector, ADFVector, state_size, param_size> tf;
       return tf(vparam, uu);
    }

    /// Evaluates the second derivative of CTD(...).
    /// Intended for internal use only.
    ADSType QEval(const Vector &vparam, ADSVector &uu)
    {
       CTD<ADSType, const Vector, ADSVector, state_size, param_size> tf;
       return tf(vparam, uu);
    }

};
}//end namespace mfem
#endif // NATIVE
#endif // ADMFEM_HPP

