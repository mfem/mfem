#ifndef ADMFEM_HPP 
#define ADMFEM_HPP

#include "mfem.hpp"
#include "fdual.hpp"
#include "tadvector.hpp"
#include "taddensemat.hpp"

namespace mfem {

template<template<typename, typename, typename, int, int, int> class CTD
         , int residual_size=1, int state_size=1, int param_size=0>
class QResidualAutoDiff
{
#ifdef MFEM_USE_CODIPACK //CoDiPack implementation
public:
private:
#else
public:
    /// MFEM native forward AD-type
    typedef ad::FDual<double> ADFType;
    typedef TADVector<ADFType> ADFVector;
    typedef TADDenseMatrix<ADFType> ADFDenseMatrix;

    /// Returns a vector valued function rr for supplied passive arguments
    /// vparam and active arguments uu. The evaluation is based on the
    /// user supplied CTD template class.
    void QResidual(const Vector &vparam, Vector &uu, Vector &rr)
    {
       CTD<double, const Vector, Vector,
               residual_size, state_size, param_size> func;
       func(vparam, uu, rr);
    }

    /// Returns the gradient of CTD(...) residual in the dense matrix jac.
    /// The dimensions of jac are residual_size x state_size, where state_size is the
    /// length of vector uu.
    void QGradResidual(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
    {
        //use native AD package
        jac.SetSize(residual_size, state_size);
        jac = 0.0;
        {
           ADFVector aduu(uu); //all dual numbers are initialized to zero
           ADFVector rr(residual_size);

           for (int ii = 0; ii < state_size; ii++)
           {
              aduu[ii].dual(1.0);
              QEval(vparam, aduu, rr);
              for (int jj = 0; jj < residual_size; jj++)
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
               residual_size, state_size, param_size> tf;
       tf(vparam, uu, rr);
    }
#endif //native MFEM implementation
};




template<template<typename, typename, typename, int, int> class CTD
         , int state_size=1, int param_size=0>
class QFunctionAutoDiff
{
#ifdef MFEM_USE_CODIPACK //CoDiPack implementation

#elif  MFEM_USE_FADBADPP //FADBAD++ implementation
    ///AD-type derived from FADBAD++
    typedef fadbad::B<double> ADFType;
    ///AD vector type for evaluation of first derivatives
    typedef TADVector<ADFType> ADFVector;
    ///AD dense matrix type for evaluation of first derivatives
    typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
    ///AD-type for the second derivatives derived from
    ///FADBAD++
    typedef fadbad::B<fadbad::F<double>> ADSType;
    /// AD vector type for evaluation of second derivatives
    typedef TADVector<ADSType> ADSVector;
    /// AD dense matrix type for  evaluation of second derivatives
    typedef TADDenseMatrix<ADSType> ADSDenseMatrix;

#else //native mfem implementation
public:
    ///MFEM native AD-type for first derivatives
    typedef ad::FDual<double> ADFType;
    typedef TADVector<ADFType> ADFVector;
    typedef TADDenseMatrix<ADFType> ADFDenseMatrix;
    ///MFEM native AD-type for second derivatives
    typedef ad::FDual<ADFType> ADSType;
    typedef TADVector<ADSType> ADSVector;
    typedef TADDenseMatrix<ADSType> ADSDenseMatrix;

    /// Evaluates a function for arguments vparam and uu.
    /// The evaluatin is based on the operator() in the
    /// user provided functor CTD.
    double QEval(const Vector &vparam, Vector &uu)
    {
        CTD<double, const Vector, Vector, state_size, param_size> tf;
        return tf(vparam,uu);
    }

    /// Returns the first derivative of CTD(...) with
    /// respect to the active arguments proved in vector uu.
    /// The length of rr is the same as for uu.
    void QResidual(const Vector &vparam, Vector &uu, Vector &rr)
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

    /// Returns the Hessian of CTD(...) in the dense matrix hh.
    /// The dimensions of jac are state_size x state_size, where state_size is the
    /// length of vector uu.
    void QGradResidual(mfem::Vector &vparam, mfem::Vector &uu, mfem::DenseMatrix &jac)
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

#endif //native AD implementation

};


}





#endif

