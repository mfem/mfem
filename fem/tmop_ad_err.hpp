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
  H1_ERROR,
  ZZ_ERROR,
  AVG_ERROR,
  ENERGY,
  GZZ_ERROR
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
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionGrad_(trueSolutionGrad)
  {};

  ~Error_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow(fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip)
  {
    dtheta_dU.SetSize(1);
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double val = 2.0* (fieldVal-trueVal);

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & /*T*/,
      const mfem::IntegrationPoint & /*ip*/)
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
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

  int Dim_ = 2;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
};

class H1Error_QoI : public QoIBaseCoefficient {
public:
  H1Error_QoI(mfem::ParGridFunction * solutionField,
              mfem::VectorCoefficient * trueSolution,
              mfem::MatrixCoefficient * trueSolutionHess)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionHess_(trueSolutionHess)
  {};

  ~H1Error_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
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

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation & /*T*/, const mfem::IntegrationPoint & /*ip*/)
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & T,
      const mfem::IntegrationPoint & ip)
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

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation & /*T*/, const mfem::IntegrationPoint &/*ip*/)
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
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

    DenseMatrix Hess;
    trueSolutionHess_->Eval(Hess, T, ip);

    grad -= trueGrad;
    Vector HessTgrad(grad.Size());
    Hess.MultTranspose(grad, HessTgrad);
    return HessTgrad;
  }
private:


  mfem::ParGridFunction * solutionField_;
  mfem::VectorCoefficient * trueSolution_;
  mfem::MatrixCoefficient * trueSolutionHess_;

  int Dim_ = 2;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;
};

class ZZError_QoI : public QoIBaseCoefficient {
public:
  ZZError_QoI(mfem::ParGridFunction * solutionField, mfem::VectorCoefficient * trueSolution)
    : solutionField_(solutionField), trueSolution_(trueSolution)
  {};

  ~ZZError_QoI() {};

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
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

  const mfem::DenseMatrix &explicitSolutionDerivative(mfem::ElementTransformation & /*T*/, const mfem::IntegrationPoint & /*ip*/)
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative(mfem::ElementTransformation & T,
      const mfem::IntegrationPoint & ip)
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

  const mfem::DenseMatrix &explicitShapeDerivative(mfem::ElementTransformation & /*T*/, const mfem::IntegrationPoint &/*ip*/)
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative(mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
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

  int Dim_ = 2;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
  mfem::DenseMatrix dUXdtheta_dGradU;
};

class AvgError_QoI : public QoIBaseCoefficient {
public:
  AvgError_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * trueSolution)
    : solutionField_(solutionField), trueSolution_(trueSolution)
  {};

  ~AvgError_QoI() {};

  double Eval( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow( fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative( mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip)
  {
    dtheta_dU.SetSize(1);

    double val = 2.0* (solutionField_->GetValue( T, ip ) - trueSolution_->Eval( T, ip ));

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative( mfem::ElementTransformation & /*T*/,
      const mfem::IntegrationPoint & /*ip*/)
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * trueSolution_;

  int Dim_ = 2;

  double theta = 0.0;
  mfem::DenseMatrix dtheta_dX;
  mfem::DenseMatrix dtheta_dU;
  mfem::DenseMatrix dtheta_dGradU;
};

class Energy_QoI : public QoIBaseCoefficient {
public:
  Energy_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * force)
    : solutionField_(solutionField), force_(force)
  {};

  ~Energy_QoI() {};

  double Eval( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double Val = force_->Eval( T, ip );

    double energy = fieldVal*Val;
    return energy;
  };

  const mfem::DenseMatrix &explicitSolutionDerivative( mfem::ElementTransformation & T, const mfem::IntegrationPoint & ip)
  {
    dtheta_dU.SetSize(1);

    double Val = force_->Eval( T, ip );
    double & matVal = dtheta_dU.Elem(0,0);
    matVal = Val;
    return dtheta_dU;
  };

  const mfem::DenseMatrix &explicitSolutionGradientDerivative( mfem::ElementTransformation & /*T*/,
      const mfem::IntegrationPoint & /*ip*/)
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const mfem::DenseMatrix &explicitShapeDerivative( mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const mfem::DenseMatrix &gradTimesexplicitSolutionGradientDerivative( mfem::ElementTransformation &T,
      const mfem::IntegrationPoint &ip)
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * force_;

  int Dim_ = 2;

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
  LFErrorDerivativeIntegrator( int IntegrationOrder = INT_MAX);
  ~LFErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  const int IntegrationOrder_;

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
  PenaltyShapeSensitivityIntegrator(mfem::Coefficient &penalty,
      const mfem::ParGridFunction &t_adjoint, int oa = 2, int ob = 2);
  void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
private:
  mfem::Coefficient *penalty_;
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

class QuantityOfInterest
{
public:
    QuantityOfInterest(mfem::ParMesh* mesh_, enum QoIType qoiType, int order_=1)
    : pmesh(mesh_), qoiType_(qoiType)
    {
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        solgf_.SetSpace(temp_fes_);

        dQdu_ = new mfem::ParLinearForm(temp_fes_);
        dQdx_ = new mfem::ParLinearForm(coord_fes_);
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

    mfem::ParMesh* pmesh;
    enum QoIType qoiType_;

    mfem::Vector X0_;
    mfem::Vector designVar;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *temp_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

    mfem::ParGridFunction solgf_;

    std::shared_ptr<QoIBaseCoefficient> ErrorCoefficient_ = nullptr;
    Array<double> gllvec_;
    int nqptsperel;

    IntegrationRules *irules;
    int quad_order;
};

class Diffusion_Solver
{
public:
    Diffusion_Solver(mfem::ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_=2, Coefficient *truesolfunc = nullptr, bool weakBC = false, VectorCoefficient *loadFuncGrad = nullptr)
    {
        weakBC_ = weakBC;
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(temp_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(temp_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(temp_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(temp_fes_);
        adjgf.SetSpace(temp_fes_);

        dQdu_ = new mfem::ParLinearForm(temp_fes_);
        dQdx_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(temp_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> temp_tdofs;
            temp_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, temp_tdofs);

            // append to global dof list
            ess_tdof_list_.Append(temp_tdofs);

            // set value in grid function
            double value = bc.second;
            ess_bc.SetSubVector(temp_tdofs, value);
        }
        bcGridFunc_.SetSpace(temp_fes_);
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
          GridFunction bdrsol(temp_fes_);
          bdrsol.ProjectBdrCoefficient(*truesolfunc, ess_bdr_attr);
          solgf.ProjectBdrCoefficient(*truesolfunc, ess_bdr_attr);
          bcGridFunc_ = bdrsol;
          trueSolCoeff = truesolfunc;
        }
        if (loadFuncGrad)
        {
          loadGradCoef_ = loadFuncGrad;
        }
    }

    ~Diffusion_Solver(){
        delete temp_fes_;
        delete coord_fes_;
        delete fec;

        delete dQdu_;
        delete dQdx_;
    }

    void UpdateMesh(mfem::Vector const &U);

    double Eval_QoI();

    void Eval_QoI_Grad();

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve();

    void ASolve( mfem::Vector & rhs );

    void SetDesign( mfem::Vector & design)
    {
        designVar = design;
    };

    void SetDesignVarFromUpdatedLocations( mfem::Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };

    void SetManufacturedSolution( mfem::Coefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    mfem::ParLinearForm * GetImplicitDqDx(){ return dQdx_; };

private:
    mfem::ParMesh* pmesh;

    mfem::Vector X0_;
    mfem::Vector designVar;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector adj;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;
    mfem::ParGridFunction adjgf;
    mfem::ParGridFunction bcGridFunc_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *temp_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    mfem::Coefficient *trueSolCoeff = nullptr;
    Array<int> ess_bdr_attr;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::Coefficient * QCoef_ = nullptr;
    mfem::VectorCoefficient *loadGradCoef_ = nullptr;

    bool weakBC_ = false;
};

class Elasticity_Solver
{
public:
    Elasticity_Solver(mfem::ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        u_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(u_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(u_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(u_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(u_fes_);
        adjgf.SetSpace(u_fes_);

        dQdu_ = new mfem::ParLinearForm(u_fes_);
        dQdx_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(u_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> u_tdofs;
            u_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);

            // set value in grid function
            double value = bc.second;
            ess_bc.SetSubVector(u_tdofs, value);
        }
        bcGridFunc_.SetSpace(u_fes_);
        bcGridFunc_.SetFromTrueDofs(ess_bc);
    }

    ~Elasticity_Solver(){
        delete u_fes_;
        delete coord_fes_;
        delete fec;

        delete dQdu_;
        delete dQdx_;
    }

    void UpdateMesh(mfem::Vector const &U);

    double Eval_QoI();

    void Eval_QoI_Grad();

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve();

    void ASolve( mfem::Vector & rhs );

    void SetDesign( mfem::Vector & design)
    {
        designVar = design;
    };

    void SetDesignVarFromUpdatedLocations( mfem::Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };

    void SetManufacturedSolution( mfem::VectorCoefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    mfem::ParLinearForm * GetImplicitDqDx(){ return dQdx_; };

private:
    mfem::ParMesh* pmesh;

    mfem::Vector X0_;
    mfem::Vector designVar;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector adj;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;
    mfem::ParGridFunction adjgf;
    mfem::ParGridFunction bcGridFunc_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *u_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::VectorCoefficient * QCoef_ = nullptr;
};

class VectorHelmholtz
{
public:
    VectorHelmholtz(mfem::ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        u_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(u_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(u_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(u_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(u_fes_);
        adjgf.SetSpace(u_fes_);

        dQdu_ = new mfem::ParLinearForm(u_fes_);
        dQdx_ = new mfem::ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        ::mfem::Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        ::mfem::Vector ess_bc(u_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            ::mfem::Array<int> u_tdofs;
            u_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);

            // set value in grid function
            double value = bc.second;
            ess_bc.SetSubVector(u_tdofs, value);
        }
        bcGridFunc_.SetSpace(u_fes_);
        bcGridFunc_.SetFromTrueDofs(ess_bc);
    }

    ~VectorHelmholtz(){
        delete u_fes_;
        delete coord_fes_;
        delete fec;

        delete dQdu_;
        delete dQdx_;
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve( mfem::Vector & rhs );

    void ASolve( mfem::Vector & rhs );

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    mfem::Vector& GetAdj(){return adj;}

    mfem::ParLinearForm * GetImplicitDqDx(){ return dQdx_; };

private:
    mfem::ParMesh* pmesh;

    mfem::Vector X0_;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector adj;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;
    mfem::ParGridFunction adjgf;
    mfem::ParGridFunction bcGridFunc_;

    mfem::ParLinearForm * dQdu_;
    mfem::ParLinearForm * dQdx_;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *u_fes_;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    mfem::Array<int> ess_tdof_list_;

    mfem::VectorCoefficient * QCoef_ = nullptr;
};

}

#endif
