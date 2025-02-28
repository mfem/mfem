#ifndef TMOP_AD_ERR_HPP
#define TMOP_AD_ERR_HPP

#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "linearform.hpp"
#include "plinearform.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "pgridfunc.hpp"
#include "pfespace.hpp"
// #include "tmop.hpp"
//#include "gslib.hpp"
#include "../linalg/mma.hpp"
#include "datacollection.hpp"


namespace mfem{
void IdentityMatrix(int dim, mfem::DenseMatrix &I);

void Vectorize(const mfem::DenseMatrix &A, mfem::Vector &a);

double MatrixInnerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B);

void ConjugationProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, const mfem::DenseMatrix &C, mfem::DenseMatrix &D);

void KroneckerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, mfem::DenseMatrix &C);

void IsotropicStiffnessMatrix(int dim, double mu, double lambda, mfem::DenseMatrix &C);

void IsotropicStiffnessMatrix3D(double E, double v, mfem::DenseMatrix &C);

void FourthOrderSymmetrizer(int dim, mfem::DenseMatrix &S);

void FourthOrderIdentity(int dim, mfem::DenseMatrix &I4);

void FourthOrderTranspose(int dim, mfem::DenseMatrix &T);

void VectorOuterProduct(const mfem::Vector &a, const mfem::Vector &b, mfem::DenseMatrix &C);
void   UnitStrain(int dim, int i, int j, mfem::DenseMatrix &E);
void   UnitStrain(int dim, int i, int j, mfem::Vector &E);
void   MatrixConjugationProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, mfem::DenseMatrix &C);


enum QoIType
{
  L2_ERROR,
  H1S_ERROR,
  ZZ_ERROR,
  AVG_ERROR,
  ENERGY,
  GZZ_ERROR,
  H1_ERROR,
  STRUC_COMPLIANCE
};

class QoIBaseCoefficient : public mfem::Coefficient {
public:
  QoIBaseCoefficient() {};

  virtual ~QoIBaseCoefficient() {};

  virtual const mfem::DenseMatrix &explicitSolutionDerivative
      (mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) = 0;

  virtual const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
    = 0;

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
    = 0;

  virtual const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) = 0;

    virtual const mfem::Vector DerivativeExactWRTX(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) {
    Vector vec(T.GetSpaceDim());
    vec = 0.0;
    return vec;
  }
private:
};

class Error_QoI : public QoIBaseCoefficient
{
public:
  Error_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * trueSolution, mfem::VectorCoefficient * trueSolutionGrad)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionGrad_(trueSolutionGrad), Dim_(trueSolutionGrad->GetVDim())
  {};

  ~Error_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow(fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double val = 2.0* (fieldVal-trueVal);

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  // 2(u-u*)(-du*/dx_a)
  virtual const mfem::Vector DerivativeExactWRTX(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector trueGrad;
    trueSolutionGrad_->Eval(trueGrad, T, ip);

    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );
    trueGrad *= -2.0*(fieldVal-trueVal);
    return trueGrad;

    // mfem::Vector grad;
    // solutionField_->GetGradient (T, grad);
    // grad *= 2.0*(fieldVal-trueVal);
    // grad += trueGrad;
    // return grad;
  }
private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * trueSolution_;
  mfem::VectorCoefficient * trueSolutionGrad_;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
};

class H1SemiError_QoI : public QoIBaseCoefficient {
public:
  H1SemiError_QoI(mfem::ParGridFunction * solutionField,
              mfem::VectorCoefficient * trueSolution,
              mfem::MatrixCoefficient * trueSolutionHess)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionHess_(trueSolutionHess), trueSolutionHessV_(nullptr), Dim_(trueSolution->GetVDim()) {};

  H1SemiError_QoI(mfem::ParGridFunction * solutionField,
              mfem::VectorCoefficient * trueSolution,
              mfem::VectorCoefficient * trueSolutionHessV)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionHess_(nullptr), trueSolutionHessV_(trueSolutionHessV), Dim_(trueSolution->GetVDim())
  {};

  ~H1SemiError_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad;
    mfem::Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    grad -= trueGrad;

    double val = grad.Norml2();
    val = val * val;

    return val;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & T,
      const mfem::IntegrationPoint & ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;
    gradMinusTrueGrad *= 2.0;

    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    dtheta_dGradU(0,0) = gradMinusTrueGrad[0];
    dtheta_dGradU(0,1) = gradMinusTrueGrad[1];


    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;
    gradMinusTrueGrad *= 2.0;

    dUXdtheta_dGradU.SetSize(Dim_, Dim_);
    dUXdtheta_dGradU = 0.0;

    dUXdtheta_dGradU(0,0) =  grad[0] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(1,0) =  grad[1] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(0,1) =  grad[0] * gradMinusTrueGrad[1];
    dUXdtheta_dGradU(1,1) =  grad[1] * gradMinusTrueGrad[1];

    dUXdtheta_dGradU.Transpose();


    return dUXdtheta_dGradU;
  };

//  -2(\grad u - \grad u*)^T.[nabla^2 u]
  virtual const mfem::Vector DerivativeExactWRTX(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad;
    mfem::Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    DenseMatrix Hess(Dim_);
    if (trueSolutionHess_)
    {
      trueSolutionHess_->Eval(Hess, T, ip);
    }
    else
    {
      Vector HessV;
      trueSolutionHessV_->Eval(HessV, T, ip);
      Hess = HessV.GetData();
    }

    grad -= trueGrad;
    Vector HessTgrad(grad.Size());
    Hess.MultTranspose(grad, HessTgrad);
    return HessTgrad;
  }

  int GetDim() { return Dim_; }
private:


  mfem::ParGridFunction * solutionField_;
  mfem::VectorCoefficient * trueSolution_;
  mfem::MatrixCoefficient * trueSolutionHess_ = nullptr;
  mfem::VectorCoefficient * trueSolutionHessV_ = nullptr;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;
};

class H1Error_QoI : public QoIBaseCoefficient {
public:
  H1Error_QoI(Error_QoI *l2error, H1SemiError_QoI *h1semierror)
    : l2error_(l2error), h1semierror_(h1semierror), Dim_(h1semierror_->GetDim())
  {};

  ~H1Error_QoI() {};

  Error_QoI *l2error_;
  H1SemiError_QoI *h1semierror_;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;
  int Dim_;
  double h1weight = 1.0/60.0;

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    return l2error_->Eval(T, ip) + h1weight*h1semierror_->Eval(T, ip);
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);
    DenseMatrix tempM(1);

    dtheta_dU = l2error_->explicitSolutionDerivative(T, ip);
    tempM = h1semierror_->explicitSolutionDerivative(T, ip);
    tempM *= h1weight;
    dtheta_dU += tempM;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    DenseMatrix tempM(1, Dim_);

    dtheta_dGradU = l2error_->explicitSolutionGradientDerivative(T, ip);
    tempM = h1semierror_->explicitSolutionGradientDerivative(T, ip);
    tempM *= h1weight;
    dtheta_dGradU += tempM;
    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    DenseMatrix tempM(Dim_, Dim_);

    dtheta_dX = l2error_->gradTimesexplicitSolutionGradientDerivative(T, ip);
    tempM = h1semierror_->gradTimesexplicitSolutionGradientDerivative(T, ip);
    tempM *= h1weight;
    dtheta_dX += tempM;

    return dtheta_dX;
  };

  // 2(u-u*)(-du*/dx_a)
  virtual const mfem::Vector DerivativeExactWRTX(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    Vector dustar_dx, temp;
    dustar_dx = l2error_->DerivativeExactWRTX(T, ip);
    temp = h1semierror_->DerivativeExactWRTX(T, ip);
    temp *= h1weight;
    dustar_dx += temp;
    return dustar_dx;
  }
};

class ZZH1Error_QoI : public QoIBaseCoefficient {
public:
  ZZH1Error_QoI(mfem::ParGridFunction * solutionField,
              mfem::VectorCoefficient * trueSolution)
    : solutionField_(solutionField), trueSolution_(trueSolution), Dim_(trueSolution->GetVDim()) {};

  ~ZZH1Error_QoI() {};

    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad;
    mfem::Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    grad -= trueGrad;

    double val = grad.Norml2();
    val = 0.5  *val * val;

    return val;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & T,
      const mfem::IntegrationPoint & ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;

    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    dtheta_dGradU(0,0) = gradMinusTrueGrad[0];
    dtheta_dGradU(0,1) = gradMinusTrueGrad[1];


    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;

    dUXdtheta_dGradU.SetSize(Dim_, Dim_);
    dUXdtheta_dGradU = 0.0;

    dUXdtheta_dGradU(0,0) =  grad[0] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(1,0) =  grad[1] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(0,1) =  grad[0] * gradMinusTrueGrad[1];
    dUXdtheta_dGradU(1,1) =  grad[1] * gradMinusTrueGrad[1];

    dUXdtheta_dGradU.Transpose();


    return dUXdtheta_dGradU;
  };

//  -2(\grad u - \grad u*)^T.[nabla^2 u]
  const mfem::Vector DerivativeExactWRTX(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    Vector HessTgrad(Dim_); HessTgrad = 0.0;
    return HessTgrad;
  };
private:

  mfem::ParGridFunction * solutionField_;
  mfem::VectorCoefficient * trueSolution_;
  mfem::MatrixCoefficient * trueSolutionHess_ = nullptr;
  mfem::VectorCoefficient * trueSolutionHessV_ = nullptr;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;

};

class ZZError_QoI : public QoIBaseCoefficient {
public:
  ZZError_QoI(mfem::ParGridFunction * solutionField, mfem::VectorCoefficient * trueSolution)
    : solutionField_(solutionField), trueSolution_(trueSolution), Dim_(trueSolution->GetVDim())
  {};

  ~ZZError_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad;
    mfem::Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    grad -= trueGrad;

    double val = grad.Norml2();
    val = val * val;

    return val;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & T,
      const mfem::IntegrationPoint & ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;
    gradMinusTrueGrad *= 2.0;

    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    dtheta_dGradU(0,0) = gradMinusTrueGrad[0];
    dtheta_dGradU(0,1) = gradMinusTrueGrad[1];


    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector grad(Dim_);
    mfem::Vector trueGrad(Dim_);
    mfem::Vector gradMinusTrueGrad(Dim_);
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);
    gradMinusTrueGrad = grad;
    gradMinusTrueGrad -= trueGrad;
    gradMinusTrueGrad *= 2.0;

    dUXdtheta_dGradU.SetSize(Dim_, Dim_);
    dUXdtheta_dGradU = 0.0;

    dUXdtheta_dGradU(0,0) =  grad[0] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(1,0) =  grad[1] * gradMinusTrueGrad[0];
    dUXdtheta_dGradU(0,1) =  grad[0] * gradMinusTrueGrad[1];
    dUXdtheta_dGradU(1,1) =  grad[1] * gradMinusTrueGrad[1];

    dUXdtheta_dGradU.Transpose();


    return dUXdtheta_dGradU;
  };
private:


  mfem::ParGridFunction * solutionField_;
  mfem::VectorCoefficient * trueSolution_;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;
};

class AvgError_QoI : public QoIBaseCoefficient {
public:
  AvgError_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * trueSolution, int Dim)
    : solutionField_(solutionField), trueSolution_(trueSolution), Dim_(Dim)
  {};

  ~AvgError_QoI() {};

  double Eval( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow( fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative( mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);

    double val = 2.0* (solutionField_->GetValue( T, ip ) - trueSolution_->Eval( T, ip ));

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * trueSolution_;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
};

class Energy_QoI : public QoIBaseCoefficient {
public:
  Energy_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * force, int Dim)
    : solutionField_(solutionField), force_(force), Dim_(Dim)
  {};

  ~Energy_QoI() {};

  double Eval( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double Val = force_->Eval( T, ip );

    double energy = fieldVal*Val;
    return energy;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative( mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);

    double Val = force_->Eval( T, ip );
    double & matVal = dtheta_dU.Elem(0,0);
    matVal = Val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * force_;

  int Dim_;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
};


class LFNodeCoordinateSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  LFNodeCoordinateSensitivityIntegrator( int IntegrationOrder = INT_MAX);
  ~LFNodeCoordinateSensitivityIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
  void SetGLLVec(Array<double> &gllvec) { gllvec_ = gllvec;}
  void SetNqptsPerEl(int nqp) { nqptsperel = nqp; }
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;

  Array<double> gllvec_;
  int nqptsperel;
};

class LFAvgErrorNodeCoordinateSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  LFAvgErrorNodeCoordinateSensitivityIntegrator(
     mfem::ParGridFunction * solutionField, GridFunctionCoefficient * elementVol,
     int IntegrationOrder = INT_MAX);
  ~LFAvgErrorNodeCoordinateSensitivityIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  mfem::ParGridFunction * solutionField_ = nullptr;
  GridFunctionCoefficient * elementVol_ = nullptr;
  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFErrorIntegrator : public mfem::LinearFormIntegrator {
public:
  LFErrorIntegrator( int IntegrationOrder = INT_MAX);
  ~LFErrorIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
    void SetGLLVec(Array<double> &gllvec) { gllvec_ = gllvec;}
    void SetNqptsPerEl(int nqp) { nqptsperel = nqp; }
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
    Array<double> gllvec_;
    int nqptsperel;
};

class LFErrorDerivativeIntegrator : public mfem::LinearFormIntegrator {
public:
  LFErrorDerivativeIntegrator( );
  ~LFErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFErrorDerivativeIntegrator_2 : public mfem::LinearFormIntegrator {
public:
  LFErrorDerivativeIntegrator_2( ParFiniteElementSpace * fespace, Array<int> count, int IntegrationOrder = INT_MAX);
  ~LFErrorDerivativeIntegrator_2() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  ParFiniteElementSpace * fespace_ = nullptr;
  Array<int> count_;
  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFFilteredFieldErrorDerivativeIntegrator : public mfem::LinearFormIntegrator {
public:
  LFFilteredFieldErrorDerivativeIntegrator( );
  ~LFFilteredFieldErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFAverageErrorDerivativeIntegrator : public mfem::LinearFormIntegrator {
public:
  LFAverageErrorDerivativeIntegrator( ParFiniteElementSpace * fespace, GridFunctionCoefficient * elementVol, int IntegrationOrder = INT_MAX);
  ~LFAverageErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  ParFiniteElementSpace * fespace_ = nullptr;
  Array<int> count_;
  const int IntegrationOrder_;

  GridFunctionCoefficient * elementVol_ = nullptr;
  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class ThermalConductivityShapeSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  ThermalConductivityShapeSensitivityIntegrator(mfem::Coefficient &conductivity, const mfem::ParGridFunction &t_primal,
      const mfem::ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *k_;
  const mfem::ParGridFunction *t_primal_;
  const mfem::ParGridFunction *t_adjoint_;
};

class ThermalConductivityShapeSensitivityIntegrator_new : public mfem::LinearFormIntegrator {
public:
  ThermalConductivityShapeSensitivityIntegrator_new(mfem::Coefficient &conductivity, const mfem::ParGridFunction &t_primal,
      const mfem::ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *k_;
  const mfem::ParGridFunction *t_primal_;
  const mfem::ParGridFunction *t_adjoint_;
};

class PenaltyMassShapeSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  PenaltyMassShapeSensitivityIntegrator(mfem::Coefficient &penalty, const mfem::ParGridFunction &t_primal,
      const mfem::ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *penalty_;
  const mfem::ParGridFunction *t_primal_;
  const mfem::ParGridFunction *t_adjoint_;
};

class PenaltyShapeSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  PenaltyShapeSensitivityIntegrator(Coefficient &t_primal, const ParGridFunction &t_adjoint, Coefficient &t_penalty, mfem::VectorCoefficient *SolGrad_= nullptr, int oa = 2, int ob = 2);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *t_primal_ = nullptr;
  mfem::Coefficient *t_penalty_ = nullptr;
  mfem::VectorCoefficient *SolGradCoeff_= nullptr;
  const mfem::ParGridFunction *t_adjoint_;
  int oa_, ob_;
};

class ThermalHeatSourceShapeSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  ThermalHeatSourceShapeSensitivityIntegrator(mfem::Coefficient &heatSource, const mfem::ParGridFunction &t_adjoint, int oa = 2,
      int ob = 2);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *Q_;
  const mfem::ParGridFunction *t_adjoint_;
  int oa_, ob_;
};

class ThermalHeatSourceShapeSensitivityIntegrator_new : public mfem::LinearFormIntegrator {
public:
  ThermalHeatSourceShapeSensitivityIntegrator_new(mfem::Coefficient &heatSource, const mfem::ParGridFunction &t_adjoint, int oa = 2,
      int ob = 2);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetLoadGrad(mfem::VectorCoefficient *LoadGrad) { LoadGrad_ = LoadGrad; };
private:
  mfem::Coefficient *Q_;
  const mfem::ParGridFunction *t_adjoint_;
  int oa_, ob_;
  mfem::VectorCoefficient *LoadGrad_;
};

class GradProjectionShapeSensitivityIntegrator : public mfem::LinearFormIntegrator {
public:
  GradProjectionShapeSensitivityIntegrator(const ParGridFunction &t_primal, const ParGridFunction &t_adjoin, VectorCoefficient & tempCoeff);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  const mfem::ParGridFunction *t_primal_;
  const mfem::ParGridFunction *t_adjoint_;
  mfem::VectorCoefficient *tempCoeff_;
};

class ElasticityStiffnessShapeSensitivityIntegrator : public mfem::LinearFormIntegrator
{
public:
    ElasticityStiffnessShapeSensitivityIntegrator(mfem::Coefficient &lambda, mfem::Coefficient &mu,
            const mfem::ParGridFunction &u_primal, const mfem::ParGridFunction &u_adjoint);
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
    mfem::Coefficient       *lambda_;
    mfem::Coefficient       *mu_;
    const mfem::ParGridFunction *u_primal_;
    const mfem::ParGridFunction *u_adjoint_;
};

class ElasticityTractionIntegrator : public mfem::LinearFormIntegrator
{
public:
    ElasticityTractionIntegrator(mfem::VectorCoefficient &f, int oa=2, int ob=2);
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
    mfem::VectorCoefficient *f_;
    int oa_, ob_;
};

class ElasticityTractionShapeSensitivityIntegrator : public mfem::LinearFormIntegrator
{
public:
    ElasticityTractionShapeSensitivityIntegrator(mfem::VectorCoefficient &f,
            const mfem::ParGridFunction &u_adjoint, int oa=2, int ob=2);
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
    mfem::VectorCoefficient *f_;
    const mfem::ParGridFunction *u_adjoint_;
    int oa_, ob_;
};


class QuantityOfInterest
{
public:
    QuantityOfInterest(mfem::ParMesh* mesh_, enum QoIType qoiType, int order_)
    : pmesh(mesh_), qoiType_(qoiType)
    {
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);
        hess_fes_ = new ParFiniteElementSpace(pmesh,fec,dim*dim);

        solgf_.SetSpace(temp_fes_);

        dQdu_ = new mfem::ParLinearForm(temp_fes_);
        dQdx_ = new mfem::ParLinearForm(coord_fes_);

        true_solgf_.SetSpace(temp_fes_);
        true_solgradgf_.SetSpace(coord_fes_);
        true_solhessgf_.SetSpace(hess_fes_);
        true_solgf_coeff_.SetGridFunction(&true_solgf_);
        true_solgradgf_coeff_.SetGridFunction(&true_solgradgf_);
        true_solhessgf_coeff_.SetGridFunction(&true_solhessgf_);

        // debug_pdc = new ParaViewDataCollection("DebugQoI", temp_fes_->GetParMesh());
        // debug_pdc->SetLevelsOfDetail(1);
        // debug_pdc->SetDataFormat(VTKFormat::BINARY);
        // debug_pdc->SetHighOrderOutput(true);
        // debug_pdc->SetCycle(0);
        // debug_pdc->SetTime(0.0);
        // debug_pdc->RegisterField("sol",&solgf_);
        // debug_pdc->RegisterField("true_sol",&true_solgf_);
        // debug_pdc->RegisterField("true_sol_grad",&true_solgradgf_);
    }

    ~QuantityOfInterest()
    {
        delete temp_fes_;
        delete coord_fes_;
        delete fec;

        delete dQdu_;
        delete dQdx_;
    }

    void setTrueSolCoeff( mfem::Coefficient * trueSolution ){ trueSolution_ = trueSolution; };
    void setTrueSolGradCoeff( mfem::VectorCoefficient * trueSolutionGrad ){ trueSolutionGrad_ = trueSolutionGrad; };
    void setTrueSolHessCoeff( mfem::MatrixCoefficient * trueSolutionHess ){ trueSolutionHess_ = trueSolutionHess; };
    void setTrueSolHessCoeff( mfem::VectorCoefficient * trueSolutionHessV ){ trueSolutionHessV_ = trueSolutionHessV; };
    void setTractionCoeff( mfem::VectorCoefficient * tractionLoad ){ tractionLoad_ = tractionLoad; }
    void SetDesign( mfem::Vector & design){ designVar = design; };
    void SetNodes( mfem::Vector & coords){ X0_ = coords; };
    void SetDesignVarFromUpdatedLocations( mfem::Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };
    void SetDiscreteSol( mfem::ParGridFunction & sol){ solgf_ = sol; };
    void UpdateMesh(mfem::Vector const &U);
    double EvalQoI();
    void EvalQoIGrad();
    mfem::ParLinearForm * GetDQDu(){ return dQdu_; };
    mfem::ParLinearForm * GetDQDx(){ return dQdx_; };
    void SetGLLVec(Array<double> &gllvec) { gllvec_ = gllvec;}
    void SetNqptsPerEl(int nqp) { nqptsperel = nqp; }
    void SetIntegrationRules(IntegrationRules *irule_, int quad_order_) { irules = irule_; quad_order = quad_order_; }
private:
    mfem::Coefficient * trueSolution_ = nullptr;
    mfem::VectorCoefficient * trueSolutionGrad_ = nullptr;
    mfem::MatrixCoefficient * trueSolutionHess_ = nullptr;
    mfem::VectorCoefficient * trueSolutionHessV_ = nullptr;

    mfem::VectorCoefficient * tractionLoad_ = nullptr;

    mfem::ParMesh* pmesh;
    enum QoIType qoiType_;

    mfem::Vector X0_;
    mfem::Vector designVar;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *temp_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;
    mfem::ParFiniteElementSpace	  *hess_fes_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

    mfem::ParGridFunction solgf_;
    mfem::ParGridFunction true_solgf_, true_solgradgf_, true_solhessgf_;
    mfem::GridFunctionCoefficient true_solgf_coeff_;
    mfem::VectorGridFunctionCoefficient true_solgradgf_coeff_;
    mfem::VectorGridFunctionCoefficient true_solhessgf_coeff_;

    ParaViewDataCollection *debug_pdc;
    int pdc_cycle = 0;

    std::shared_ptr<QoIBaseCoefficient> ErrorCoefficient_ = nullptr;
    Array<double> gllvec_;
    int nqptsperel;

    IntegrationRules *irules;
    int quad_order;
};

class PhysicsSolverBase
{
  public:
    PhysicsSolverBase( mfem::ParMesh* mesh_, int order_)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        dQdx_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();
    };

    virtual ~PhysicsSolverBase()
    {
        delete physics_fes_;
        delete coord_fes_;
        delete fec;

        delete dQdu_;
        delete dQdx_;
    };

    void UpdateMesh(mfem::Vector const &U);

    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    virtual void FSolve() = 0;

    virtual void ASolve( mfem::Vector & rhs ) = 0;

    void SetDesign( mfem::Vector & design)
    {
        designVar = design;
    };

    void SetDesignVarFromUpdatedLocations( mfem::Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    mfem::ParLinearForm * GetImplicitDqDx(){ return dQdx_; };

  protected:
    mfem::ParMesh* pmesh;

    mfem::Vector X0_;
    mfem::Vector designVar;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *physics_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector adj;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;
    mfem::ParGridFunction adjgf;
    mfem::ParGridFunction bcGridFunc_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

        //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;
};

class Diffusion_Solver : public PhysicsSolverBase
{
public:
    Diffusion_Solver(mfem::ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_, Coefficient *truesolfunc = nullptr, bool weakBC = false, VectorCoefficient *loadFuncGrad = nullptr)
    : PhysicsSolverBase(mesh_, order_)
    {
        weakBC_ = weakBC;

        physics_fes_ = new ParFiniteElementSpace(pmesh,fec);

        sol.SetSize(physics_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(physics_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(physics_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(physics_fes_);
        adjgf.SetSpace(physics_fes_);

        dQdu_ = new mfem::ParLinearForm(physics_fes_);  

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(physics_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> temp_tdofs;
            physics_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, temp_tdofs);

            // append to global dof list
            ess_tdof_list_.Append(temp_tdofs);

            // set value in grid function
            double value = bc.second;
            ess_bc.SetSubVector(temp_tdofs, value);
        }
        bcGridFunc_.SetSpace(physics_fes_);
        bcGridFunc_.SetFromTrueDofs(ess_bc);

        ess_bdr_attr.SetSize(maxAttribute);
        ess_bdr_attr = 0;
        for (const auto &bc: ess_bdr)
        {
          int attribute = bc.first;
          ess_bdr_attr[attribute-1] = 1;
        }
        if (truesolfunc)
        {
          bcGridFunc_ = 0.0;
          GridFunction bdrsol(physics_fes_);
          bdrsol.ProjectBdrCoefficient(*truesolfunc, ess_bdr_attr);
          solgf.ProjectBdrCoefficient(*truesolfunc, ess_bdr_attr);
          bcGridFunc_ = bdrsol;
          trueSolCoeff = truesolfunc;
        }
        if (loadFuncGrad)
        {
          loadGradCoef_ = loadFuncGrad;
        }

        trueloadgradgf_.SetSpace(coord_fes_);
        trueloadgradgf_coeff_.SetGridFunction(&trueloadgradgf_);
    }

    ~Diffusion_Solver(){
    }

    /// Solves the forward problem.
    void FSolve() override ;

    void ASolve( mfem::Vector & rhs ) override ;

    void SetManufacturedSolution( mfem::Coefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

    void setTrueSolGradCoeff( mfem::VectorCoefficient * trueSolutionGradCoef ){ trueSolutionGradCoef_ = trueSolutionGradCoef; };

private:

    mfem::Coefficient *trueSolCoeff = nullptr;
    Array<int> ess_bdr_attr;

    int pdc_cycle = 0;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::Coefficient * QCoef_ = nullptr;
    mfem::VectorCoefficient *loadGradCoef_ = nullptr;
    mfem::VectorCoefficient *trueSolutionGradCoef_ = nullptr;

    mfem::ParGridFunction trueloadgradgf_;
    mfem::VectorGridFunctionCoefficient trueloadgradgf_coeff_;

    bool weakBC_ = false;
};

class Elasticity_Solver : public PhysicsSolverBase
{
public:
    Elasticity_Solver(mfem::ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_)
    : PhysicsSolverBase( mesh_, order_ )
    {
        int dim=pmesh->Dimension();
        physics_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(physics_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(physics_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(physics_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(physics_fes_);
        adjgf.SetSpace(physics_fes_);

        dQdu_ = new mfem::ParLinearForm(physics_fes_);

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(physics_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> u_tdofs;
            physics_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);

            // set value in grid function
            double value = bc.second;
            ess_bc.SetSubVector(u_tdofs, value);
        }
        bcGridFunc_.SetSpace(physics_fes_);
        bcGridFunc_.SetFromTrueDofs(ess_bc);
    }

    ~Elasticity_Solver(){
    }

    /// Solves the forward problem.
    void FSolve() override ;

    void ASolve( mfem::Vector & rhs ) override ;

    void SetLoad( mfem::VectorCoefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

private:

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::VectorCoefficient * QCoef_ = nullptr;
};

class VectorHelmholtz
{
public:
    VectorHelmholtz(mfem::ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, real_t radius, int order_)
    {

        radius_ = new ConstantCoefficient(radius);
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        // sol.SetSize(coord_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(coord_fes_->GetTrueVSize()); rhs=0.0;
        // adj.SetSize(coord_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(coord_fes_);
        // adjgf.SetSpace(coord_fes_);

        dQdx_ = new mfem::ParLinearForm(coord_fes_);
        dQdu_ = new mfem::ParLinearForm(temp_fes_);
        dQdxshape_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(coord_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;
            int component = bc.second;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> u_tdofs;
            coord_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs, component);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);
        }
    }

    VectorHelmholtz(mfem::ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, ProductCoefficient *radius, int order_)
    {
        pradius_ = radius;
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        // sol.SetSize(coord_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(coord_fes_->GetTrueVSize()); rhs=0.0;
        // adj.SetSize(coord_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(coord_fes_);
        // adjgf.SetSpace(coord_fes_);

        dQdx_ = new mfem::ParLinearForm(coord_fes_);
        dQdu_ = new mfem::ParLinearForm(temp_fes_);
        dQdxshape_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(coord_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;
            int component = bc.second;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> u_tdofs;
            coord_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs, component);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);
        }
    }

    ~VectorHelmholtz(){
        delete coord_fes_;
        delete temp_fes_;
        delete fec;

        delete dQdx_;
        delete radius_;
        delete dQdu_;
        delete dQdxshape_;

        delete QGF_;
        delete QCoef_;
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve( );

    void ASolve( mfem::Vector & rhs, bool isGradX = true );

    void setLoadGridFunction( Vector & loadGF)
    {
        if(coeffSet) { mfem_error("coeff already set"); }
        GFSet = true;
        delete QGF_;
        delete QCoef_;
        QGF_ = new mfem::ParGridFunction(coord_fes_);
        *QGF_ = loadGF;
        // QGF_->SetFromTrueDofs(loadGF);
        QCoef_ = new VectorGridFunctionCoefficient(QGF_);
    };

    void setLoadCoeff(mfem::VectorCoefficient * loadCoeff)
    {
      if(coeffSet) { mfem_error("coeff already set"); }
      coeffSet = true;
      QCoef_ =loadCoeff; };

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    mfem::Vector& GetSolutionVec(){return solgf;}
    mfem::Vector GetSolutionTVec(){
      solgf.SetTrueVector();
      return solgf.GetTrueVector();}

    /// Returns the adjoint solution vector.
    // mfem::Vector& GetAdj(){return adj;}

    mfem::ParLinearForm * GetImplicitDqDx(){ return dQdx_; };
    mfem::Vector GetImplicitDqDxVec(){ return *dQdx_; };

    mfem::ParLinearForm * GetImplicitDqDxshape(){ return dQdxshape_; };

    mfem::ParLinearForm * GetImplicitDqDu(){ return dQdu_; };

private:
    mfem::ParMesh* pmesh;

    mfem::Vector X0_;

    //solution true vector
    // mfem::Vector sol;
    // mfem::Vector adj;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;
    // mfem::ParGridFunction adjgf;
    mfem::ParGridFunction bcGridFunc_;

    mfem::ParLinearForm * dQdx_;
    mfem::ParLinearForm * dQdxshape_;
    mfem::ParLinearForm * dQdu_;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *temp_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::ParGridFunction* QGF_ = nullptr;
    mfem::VectorCoefficient * QCoef_ = nullptr;

    Coefficient * radius_;
    ProductCoefficient *pradius_ = nullptr;

    bool GFSet = false;
    bool coeffSet = false;
};
}
#endif
