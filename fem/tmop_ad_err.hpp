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
void IdentityMatrix(int dim, DenseMatrix &I);

void Vectorize(const DenseMatrix &A, Vector &a);

double MatrixInnerProduct(const DenseMatrix &A, const DenseMatrix &B);

void ConjugationProduct(const DenseMatrix &A, const DenseMatrix &B, const DenseMatrix &C, DenseMatrix &D);

void KroneckerProduct(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);

void IsotropicStiffnessMatrix(int dim, double mu, double lambda, DenseMatrix &C);

void IsotropicStiffnessMatrix3D(double E, double v, DenseMatrix &C);

void FourthOrderSymmetrizer(int dim, DenseMatrix &S);

void FourthOrderIdentity(int dim, DenseMatrix &I4);

void FourthOrderTranspose(int dim, DenseMatrix &T);

void VectorOuterProduct(const Vector &a, const Vector &b, DenseMatrix &C);
void   UnitStrain(int dim, int i, int j, DenseMatrix &E);
void   UnitStrain(int dim, int i, int j, Vector &E);
void   MatrixConjugationProduct(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);

class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient * lambda=nullptr;
   Coefficient * mu=nullptr;
   GridFunction *u = nullptr; // displacement
   DenseMatrix grad; // auxiliary matrix, used in Eval

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction * u_)
      : lambda(lambda_), mu(mu_),  u(u_)
   {
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
    MFEM_ASSERT(u, "displacement field is not set");
      real_t L = lambda->Eval(T, ip);
      real_t M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      real_t div_u = grad.Trace();
      int dim = T.GetSpaceDim();

      DenseMatrix I;
      DenseMatrix stress(dim);

      IdentityMatrix(dim, I);
      I *= L * div_u;

      grad.Symmetrize();
      stress= grad;
      stress *= 2.0*M;
      stress += I;

      return MatrixInnerProduct(grad, stress);;
   }

   void SetU(GridFunction *u_) { u = u_; }
};


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

class QoIBaseCoefficient : public Coefficient {
public:
  QoIBaseCoefficient() {};

  virtual ~QoIBaseCoefficient() {};

  virtual const DenseMatrix &explicitSolutionDerivative
      (ElementTransformation &T, const IntegrationPoint &ip) = 0;

  virtual const DenseMatrix &explicitSolutionGradientDerivative(ElementTransformation &T,
      const IntegrationPoint &ip)
    = 0;

  virtual const DenseMatrix &gradTimesexplicitSolutionGradientDerivative(ElementTransformation &T,
      const IntegrationPoint &ip)
    = 0;

  virtual const DenseMatrix &explicitShapeDerivative(ElementTransformation &T, const IntegrationPoint &ip) = 0;

    virtual const Vector DerivativeExactWRTX(ElementTransformation &T, const IntegrationPoint &ip) {
    Vector vec(T.GetSpaceDim());
    vec = 0.0;
    return vec;
  }
private:
};

class Error_QoI : public QoIBaseCoefficient
{
public:
  Error_QoI(ParGridFunction * solutionField, Coefficient * trueSolution, VectorCoefficient * trueSolutionGrad)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionGrad_(trueSolutionGrad), Dim_(trueSolutionGrad->GetVDim())
  {};

  ~Error_QoI() {};

  double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow(fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const DenseMatrix &explicitSolutionDerivative(ElementTransformation & T, const IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double val = 2.0* (fieldVal-trueVal);

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const DenseMatrix &explicitSolutionGradientDerivative(ElementTransformation &T,
      const IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const DenseMatrix &explicitShapeDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const DenseMatrix &gradTimesexplicitSolutionGradientDerivative(ElementTransformation &T,
      const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  // 2(u-u*)(-du*/dx_a)
  virtual const Vector DerivativeExactWRTX(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector trueGrad;
    trueSolutionGrad_->Eval(trueGrad, T, ip);

    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );
    trueGrad *= -2.0*(fieldVal-trueVal);
    return trueGrad;

    // Vector grad;
    // solutionField_->GetGradient (T, grad);
    // grad *= 2.0*(fieldVal-trueVal);
    // grad += trueGrad;
    // return grad;
  }
private:

  ParGridFunction * solutionField_;
  Coefficient * trueSolution_;
  VectorCoefficient * trueSolutionGrad_;

  int Dim_;

  double theta = 0.0;
  DenseMatrix dtheta_dX;
  DenseMatrix dtheta_dU;
  DenseMatrix dtheta_dGradU;
};

class H1SemiError_QoI : public QoIBaseCoefficient {
public:
  H1SemiError_QoI(ParGridFunction * solutionField,
              VectorCoefficient * trueSolution,
              MatrixCoefficient * trueSolutionHess)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionHess_(trueSolutionHess), trueSolutionHessV_(nullptr), Dim_(trueSolution->GetVDim()) {};

  H1SemiError_QoI(ParGridFunction * solutionField,
              VectorCoefficient * trueSolution,
              VectorCoefficient * trueSolutionHessV)
    : solutionField_(solutionField), trueSolution_(trueSolution), trueSolutionHess_(nullptr), trueSolutionHessV_(trueSolutionHessV), Dim_(trueSolution->GetVDim())
  {};

  ~H1SemiError_QoI() {};

  double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector grad;
    Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    grad -= trueGrad;

    double val = grad.Norml2();
    val = val * val;

    return val;
  };

  const DenseMatrix &explicitSolutionDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const DenseMatrix &explicitSolutionGradientDerivative(ElementTransformation & T,
      const IntegrationPoint & ip) override
  {
    Vector grad(Dim_);
    Vector trueGrad(Dim_);
    Vector gradMinusTrueGrad(Dim_);
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

  const DenseMatrix &explicitShapeDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const DenseMatrix &gradTimesexplicitSolutionGradientDerivative(ElementTransformation &T,
      const IntegrationPoint &ip) override
  {
    Vector grad(Dim_);
    Vector trueGrad(Dim_);
    Vector gradMinusTrueGrad(Dim_);
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
  virtual const Vector DerivativeExactWRTX(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector grad;
    Vector trueGrad;
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


  ParGridFunction * solutionField_;
  VectorCoefficient * trueSolution_;
  MatrixCoefficient * trueSolutionHess_ = nullptr;
  VectorCoefficient * trueSolutionHessV_ = nullptr;

  int Dim_;

  double theta = 0.0;
  DenseMatrix dtheta_dX;
  DenseMatrix dtheta_dU;
  DenseMatrix dtheta_dGradU;
  DenseMatrix dUXdtheta_dGradU;
};

class GZZError_QoI : public QoIBaseCoefficient {
public:
  GZZError_QoI(ParGridFunction * solutionField,
              VectorCoefficient * trueSolution)
    : solutionField_(solutionField), trueSolution_(trueSolution), Dim_(trueSolution->GetVDim()) {};

  ~GZZError_QoI() {};

    double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector grad;
    Vector trueGrad;
    trueSolution_->Eval (trueGrad, T, ip);
    solutionField_->GetGradient (T, grad);

    grad -= trueGrad;

    double val = grad.Norml2();
    val = 0.5  *val * val;

    return val;
  };

  const DenseMatrix &explicitSolutionDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dU.SetSize(1);
    dtheta_dU = 0.0;

    return dtheta_dU;
  };

  const DenseMatrix &explicitSolutionGradientDerivative(ElementTransformation & T,
      const IntegrationPoint & ip) override
  {
    Vector grad(Dim_);
    Vector trueGrad(Dim_);
    Vector gradMinusTrueGrad(Dim_);
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

  const DenseMatrix &explicitShapeDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  const DenseMatrix &gradTimesexplicitSolutionGradientDerivative(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector grad(Dim_);
    Vector trueGrad(Dim_);
    Vector gradMinusTrueGrad(Dim_);
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
  const Vector DerivativeExactWRTX(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector HessTgrad(Dim_); HessTgrad = 0.0;
    return HessTgrad;
  };
private:

  ParGridFunction * solutionField_;
  VectorCoefficient * trueSolution_;
  MatrixCoefficient * trueSolutionHess_ = nullptr;
  VectorCoefficient * trueSolutionHessV_ = nullptr;

  int Dim_;

  double theta = 0.0;
  DenseMatrix dtheta_dX;
  DenseMatrix dtheta_dU;
  DenseMatrix dtheta_dGradU;
  DenseMatrix dUXdtheta_dGradU;

};

class AvgError_QoI : public QoIBaseCoefficient {
public:
  AvgError_QoI(ParGridFunction * solutionField, Coefficient * trueSolution, int Dim)
    : solutionField_(solutionField), trueSolution_(trueSolution), Dim_(Dim)
  {};

  ~AvgError_QoI() {};

  double Eval( ElementTransformation &T, const IntegrationPoint &ip) override
  {
    double fieldVal = solutionField_->GetValue( T, ip );
    double trueVal = trueSolution_->Eval( T, ip );

    double squaredError = std::pow( fieldVal-trueVal, 2.0);

    return squaredError;
  };

  const DenseMatrix &explicitSolutionDerivative( ElementTransformation & T, const IntegrationPoint & ip) override
  {
    dtheta_dU.SetSize(1);

    double val = 2.0* (solutionField_->GetValue( T, ip ) - trueSolution_->Eval( T, ip ));

    double & matVal = dtheta_dU.Elem(0,0);
    matVal = val;
    return dtheta_dU;
  };

  const DenseMatrix &explicitSolutionGradientDerivative( ElementTransformation &T,
      const IntegrationPoint &ip) override
  {
    dtheta_dGradU.SetSize(1, Dim_);
    dtheta_dGradU = 0.0;

    return dtheta_dGradU;
  };

  const DenseMatrix &explicitShapeDerivative( ElementTransformation &T, const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(1, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

  virtual const DenseMatrix &gradTimesexplicitSolutionGradientDerivative( ElementTransformation &T,
      const IntegrationPoint &ip) override
  {
    dtheta_dX.SetSize(Dim_, Dim_);
    dtheta_dX = 0.0;

    return dtheta_dX;
  };

private:

  ParGridFunction * solutionField_;
  Coefficient * trueSolution_;

  int Dim_;

  double theta = 0.0;
  DenseMatrix dtheta_dX;
  DenseMatrix dtheta_dU;
  DenseMatrix dtheta_dGradU;
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

class LFNodeCoordinateSensitivityIntegrator : public LinearFormIntegrator {
public:
  LFNodeCoordinateSensitivityIntegrator( int IntegrationOrder = INT_MAX);
  ~LFNodeCoordinateSensitivityIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
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

class LFAvgErrorNodeCoordinateSensitivityIntegrator : public LinearFormIntegrator {
public:
  LFAvgErrorNodeCoordinateSensitivityIntegrator(
     ParGridFunction * solutionField, GridFunctionCoefficient * elementVol,
     int IntegrationOrder = INT_MAX);
  ~LFAvgErrorNodeCoordinateSensitivityIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  ParGridFunction * solutionField_ = nullptr;
  GridFunctionCoefficient * elementVol_ = nullptr;
  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFErrorIntegrator : public LinearFormIntegrator {
public:
  LFErrorIntegrator( int IntegrationOrder = INT_MAX);
  ~LFErrorIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
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

class LFErrorDerivativeIntegrator : public LinearFormIntegrator {
public:
  LFErrorDerivativeIntegrator( );
  ~LFErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFErrorDerivativeIntegrator_2 : public LinearFormIntegrator {
public:
  LFErrorDerivativeIntegrator_2( ParFiniteElementSpace * fespace, Array<int> count, int IntegrationOrder = INT_MAX);
  ~LFErrorDerivativeIntegrator_2() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  ParFiniteElementSpace * fespace_ = nullptr;
  Array<int> count_;
  const int IntegrationOrder_;

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFFilteredFieldErrorDerivativeIntegrator : public LinearFormIntegrator {
public:
  LFFilteredFieldErrorDerivativeIntegrator( );
  ~LFFilteredFieldErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class LFAverageErrorDerivativeIntegrator : public LinearFormIntegrator {
public:
  LFAverageErrorDerivativeIntegrator( ParFiniteElementSpace * fespace, GridFunctionCoefficient * elementVol, int IntegrationOrder = INT_MAX);
  ~LFAverageErrorDerivativeIntegrator() {};
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetQoI(std::shared_ptr<QoIBaseCoefficient> QoI) { QoI_ = QoI; };
private:
  std::shared_ptr<QoIBaseCoefficient> QoIFactoryFunction(const int dim);

  ParFiniteElementSpace * fespace_ = nullptr;
  Array<int> count_;
  const int IntegrationOrder_;

  GridFunctionCoefficient * elementVol_ = nullptr;
  std::shared_ptr<QoIBaseCoefficient> QoI_ = nullptr;
};

class ThermalConductivityShapeSensitivityIntegrator : public LinearFormIntegrator {
public:
  ThermalConductivityShapeSensitivityIntegrator(Coefficient &conductivity, const ParGridFunction &t_primal,
      const ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  Coefficient *k_;
  const ParGridFunction *t_primal_;
  const ParGridFunction *t_adjoint_;
};

class ThermalConductivityShapeSensitivityIntegrator_new : public LinearFormIntegrator {
public:
  ThermalConductivityShapeSensitivityIntegrator_new(Coefficient &conductivity, const ParGridFunction &t_primal,
      const ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  Coefficient *k_;
  const ParGridFunction *t_primal_;
  const ParGridFunction *t_adjoint_;
};

class PenaltyMassShapeSensitivityIntegrator : public LinearFormIntegrator {
public:
  PenaltyMassShapeSensitivityIntegrator(Coefficient &penalty, const ParGridFunction &t_primal,
      const ParGridFunction &t_adjoint);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  Coefficient *penalty_;
  const ParGridFunction *t_primal_;
  const ParGridFunction *t_adjoint_;
};

class PenaltyShapeSensitivityIntegrator : public LinearFormIntegrator {
public:
  PenaltyShapeSensitivityIntegrator(Coefficient &t_primal, const ParGridFunction &t_adjoint, Coefficient &t_penalty, VectorCoefficient *SolGrad_= nullptr, int oa = 2, int ob = 2);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  Coefficient *t_primal_ = nullptr;
  Coefficient *t_penalty_ = nullptr;
  VectorCoefficient *SolGradCoeff_= nullptr;
  const ParGridFunction *t_adjoint_;
  int oa_, ob_;
};

class ThermalHeatSourceShapeSensitivityIntegrator : public LinearFormIntegrator {
public:
  ThermalHeatSourceShapeSensitivityIntegrator(Coefficient &heatSource, const ParGridFunction &t_adjoint, int oa = 2,
      int ob = 2);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  Coefficient *Q_;
  const ParGridFunction *t_adjoint_;
  int oa_, ob_;
};

class ThermalHeatSourceShapeSensitivityIntegrator_new : public LinearFormIntegrator {
public:
  ThermalHeatSourceShapeSensitivityIntegrator_new(Coefficient &heatSource, const ParGridFunction &t_adjoint, int oa = 2,
      int ob = 2);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
  void SetLoadGrad(VectorCoefficient *LoadGrad) { LoadGrad_ = LoadGrad; };
private:
  Coefficient *Q_;
  const ParGridFunction *t_adjoint_;
  int oa_, ob_;
  VectorCoefficient *LoadGrad_;
};

class GradProjectionShapeSensitivityIntegrator : public LinearFormIntegrator {
public:
  GradProjectionShapeSensitivityIntegrator(const ParGridFunction &t_primal, const ParGridFunction &t_adjoin, VectorCoefficient & tempCoeff);
  void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
  const ParGridFunction *t_primal_;
  const ParGridFunction *t_adjoint_;
  VectorCoefficient *tempCoeff_;
};

class ElasticityStiffnessShapeSensitivityIntegrator : public LinearFormIntegrator
{
public:
    ElasticityStiffnessShapeSensitivityIntegrator(Coefficient &lambda, Coefficient &mu,
            const ParGridFunction &u_primal, const ParGridFunction &u_adjoint);
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
    Coefficient       *lambda_;
    Coefficient       *mu_;
    const ParGridFunction *u_primal_;
    const ParGridFunction *u_adjoint_;
};

class ElasticityTractionIntegrator : public LinearFormIntegrator
{
public:
    ElasticityTractionIntegrator(VectorCoefficient &f, int oa=2, int ob=2);
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
    VectorCoefficient *f_;
    int oa_, ob_;
};

class ElasticityTractionShapeSensitivityIntegrator : public LinearFormIntegrator
{
public:
    ElasticityTractionShapeSensitivityIntegrator(VectorCoefficient &f,
            const ParGridFunction &u_adjoint, int oa=2, int ob=2);
    void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T, Vector &elvect);
private:
    VectorCoefficient *f_;
    const ParGridFunction *u_adjoint_;
    int oa_, ob_;
};


class QuantityOfInterest
{
public:
    QuantityOfInterest(ParMesh* mesh_, enum QoIType qoiType, int order_, Array<int> NeumannBdr = {} ,int physicsdim = 1)
    : pmesh(mesh_), qoiType_(qoiType), bdr(NeumannBdr)
    {
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,fec,physicsdim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);
        hess_fes_ = new ParFiniteElementSpace(pmesh,fec,dim*dim);

        solgf_.SetSpace(temp_fes_);

        dQdu_ = new ParLinearForm(temp_fes_);
        dQdx_ = new ParLinearForm(coord_fes_);

        if(physicsdim ==1)
        {
          true_solgf_.SetSpace(temp_fes_);
          true_solgradgf_.SetSpace(coord_fes_);
          true_solhessgf_.SetSpace(hess_fes_);
          true_solgf_coeff_.SetGridFunction(&true_solgf_);
          true_solgradgf_coeff_.SetGridFunction(&true_solgradgf_);
          true_solhessgf_coeff_.SetGridFunction(&true_solhessgf_);
        }

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

    void setTrueSolCoeff( Coefficient * trueSolution ){ trueSolution_ = trueSolution; };
    void setTrueSolGradCoeff( VectorCoefficient * trueSolutionGrad ){ trueSolutionGrad_ = trueSolutionGrad; };
    void setTrueSolHessCoeff( MatrixCoefficient * trueSolutionHess ){ trueSolutionHess_ = trueSolutionHess; };
    void setTrueSolHessCoeff( VectorCoefficient * trueSolutionHessV ){ trueSolutionHessV_ = trueSolutionHessV; };
    void setTractionCoeff( VectorCoefficient * tractionLoad ){ tractionLoad_ = tractionLoad; }
    void SetDesign( Vector & design){ designVar = design; };
    void SetNodes( Vector & coords){ X0_ = coords; };
    void SetDesignVarFromUpdatedLocations( Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };
    void SetDiscreteSol( ParGridFunction & sol){ solgf_ = sol; };
    void UpdateMesh(Vector const &U);
    double EvalQoI();
    void EvalQoIGrad();
    ParLinearForm * GetDQDu(){ return dQdu_; };
    ParLinearForm * GetDQDx(){ return dQdx_; };
    void SetGLLVec(Array<double> &gllvec) { gllvec_ = gllvec;}
    void SetNqptsPerEl(int nqp) { nqptsperel = nqp; }
    void SetIntegrationRules(IntegrationRules *irule_, int quad_order_) { irules = irule_; quad_order = quad_order_; }
    Coefficient * GetTrueSolCoeff() { return trueSolution_; }
    VectorCoefficient *GetTrueSolGradCoeff() { return trueSolutionGrad_; }
private:
    Coefficient * trueSolution_ = nullptr;
    VectorCoefficient * trueSolutionGrad_ = nullptr;
    MatrixCoefficient * trueSolutionHess_ = nullptr;
    VectorCoefficient * trueSolutionHessV_ = nullptr;

    VectorCoefficient * tractionLoad_ = nullptr;

    ParMesh* pmesh;
    enum QoIType qoiType_;

    Vector X0_;
    Vector designVar;

    FiniteElementCollection *fec;
    ParFiniteElementSpace	  *temp_fes_;
    ParFiniteElementSpace	  *coord_fes_;
    ParFiniteElementSpace	  *hess_fes_;

    ParLinearForm * dQdu_;
    ParLinearForm * dQdx_;

    ParGridFunction solgf_;
    ParGridFunction true_solgf_, true_solgradgf_, true_solhessgf_;
    GridFunctionCoefficient true_solgf_coeff_;
    VectorGridFunctionCoefficient true_solgradgf_coeff_;
    VectorGridFunctionCoefficient true_solhessgf_coeff_;

    ParaViewDataCollection *debug_pdc;
    int pdc_cycle = 0;

    std::shared_ptr<QoIBaseCoefficient> ErrorCoefficient_ = nullptr;
    Array<double> gllvec_;
    int nqptsperel;

    IntegrationRules *irules;
    int quad_order;

    Array<int> bdr;
};

class PhysicsSolverBase
{
  public:
    PhysicsSolverBase( ParMesh* mesh_, int order_)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        dQdx_ = new ParLinearForm(coord_fes_);

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

    void UpdateMesh(Vector const &U);

    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=2000)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    virtual void FSolve() = 0;

    virtual void ASolve( Vector & rhs ) = 0;

    void SetDesign( Vector & design)
    {
        designVar = design;
    };

    void SetDesignVarFromUpdatedLocations( Vector & design)
    {
        designVar = design;
        designVar -= X0_;
    };

    /// Returns the solution
    ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    Vector& GetSol(){return sol;}

    /// Returns the adjoint solution vector.
    Vector& GetAdj(){return adj;}

    ParLinearForm * GetImplicitDqDx(){ return dQdx_; };

  protected:
    ParMesh* pmesh;

    Vector X0_;
    Vector designVar;

    FiniteElementCollection *fec;
    ParFiniteElementSpace	  *physics_fes_;
    ParFiniteElementSpace	  *coord_fes_;

    //solution true vector
    Vector sol;
    Vector adj;
    Vector rhs;
    ParGridFunction solgf;
    ParGridFunction adjgf;
    ParGridFunction bcGridFunc_;

    ParLinearForm * dQdu_;
    ParLinearForm * dQdx_;

        //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;
};

class Diffusion_Solver : public PhysicsSolverBase
{
public:
    Diffusion_Solver(ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, int order_, Coefficient *truesolfunc = nullptr, bool weakBC = false, VectorCoefficient *loadFuncGrad = nullptr)
    : PhysicsSolverBase(mesh_, order_)
    {
        weakBC_ = weakBC;

        physics_fes_ = new ParFiniteElementSpace(pmesh,fec);

        sol.SetSize(physics_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(physics_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(physics_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(physics_fes_);
        adjgf.SetSpace(physics_fes_);

        dQdu_ = new ParLinearForm(physics_fes_);

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        Vector ess_bc(physics_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            Array<int> temp_tdofs;
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

    void ASolve( Vector & rhs ) override ;

    void SetManufacturedSolution( Coefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

    void setTrueSolGradCoeff( VectorCoefficient * trueSolutionGradCoef ){ trueSolutionGradCoef_ = trueSolutionGradCoef; };

private:

    Coefficient *trueSolCoeff = nullptr;
    Array<int> ess_bdr_attr;

    int pdc_cycle = 0;

    // holds NBC in coefficient form
    std::map<int, Coefficient*> ncc;

    Array<int> ess_tdof_list_;

    Coefficient * QCoef_ = nullptr;
    VectorCoefficient *loadGradCoef_ = nullptr;
    VectorCoefficient *trueSolutionGradCoef_ = nullptr;

    ParGridFunction trueloadgradgf_;
    VectorGridFunctionCoefficient trueloadgradgf_coeff_;

    bool weakBC_ = false;
};

class Elasticity_Solver : public PhysicsSolverBase
{
public:
    Elasticity_Solver(ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, const Array<int> & neumannBdr, int order_)
    : PhysicsSolverBase( mesh_, order_ ), bdr(neumannBdr)
    {
        int dim=pmesh->Dimension();
        physics_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        sol.SetSize(physics_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(physics_fes_->GetTrueVSize()); rhs=0.0;
        adj.SetSize(physics_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(physics_fes_);
        adjgf.SetSpace(physics_fes_);

        dQdu_ = new ParLinearForm(physics_fes_);

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        Vector ess_bc(physics_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            Array<int> u_tdofs;
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

    void ASolve( Vector & rhs ) override ;

    void SetLoad( VectorCoefficient * QCoef )
    {
      QCoef_ = QCoef;
    }

private:

    // holds NBC in coefficient form
    std::map<int, Coefficient*> ncc;

    Array<int> ess_tdof_list_;

    VectorCoefficient * QCoef_ = nullptr;

    Array<int> bdr;
};

class VectorHelmholtz
{
public:
    VectorHelmholtz(ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, real_t radius, int order_)
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

        dQdx_ = new ParLinearForm(coord_fes_);
        dQdu_ = new ParLinearForm(temp_fes_);
        dQdxshape_ = new ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        Vector ess_bc(coord_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;
            int component = bc.second;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            Array<int> u_tdofs;
            coord_fes_->GetEssentialTrueDofs(bdr_attr_is_ess, u_tdofs, component);

            // append to global dof list
            ess_tdof_list_.Append(u_tdofs);
        }
    }

    VectorHelmholtz(ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, ProductCoefficient *radius, int order_)
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

        dQdx_ = new ParLinearForm(coord_fes_);
        dQdu_ = new ParLinearForm(temp_fes_);
        dQdxshape_ = new ParLinearForm(coord_fes_);

        SetLinearSolver();

        // store list of essential dofs
        int maxAttribute = pmesh->bdr_attributes.Max();
        Array<int> bdr_attr_is_ess(maxAttribute);
        ess_tdof_list_.DeleteAll();
        Vector ess_bc(coord_fes_->GetTrueVSize());
        ess_bc = 0.0;

        // loop over input attribute, value pairs
        for (const auto &bc: ess_bdr)
        {
            int attribute = bc.first;
            int component = bc.second;

            // get dofs associated with this attribute, component pair
            bdr_attr_is_ess = 0;
            bdr_attr_is_ess[attribute - 1] = 1; // mfem attributes 1-indexed, arrays 0-indexed
            Array<int> u_tdofs;
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

    void ASolve( Vector & rhs, bool isGradX = true );

    void setLoadGridFunction( Vector & loadGF)
    {
        if(coeffSet) { mfem_error("coeff already set"); }
        GFSet = true;
        delete QGF_;
        delete QCoef_;
        QGF_ = new ParGridFunction(coord_fes_);
        *QGF_ = loadGF;
        // QGF_->SetFromTrueDofs(loadGF);
        QCoef_ = new VectorGridFunctionCoefficient(QGF_);
    };

    void setLoadCoeff(VectorCoefficient * loadCoeff)
    {
      if(coeffSet) { mfem_error("coeff already set"); }
      coeffSet = true;
      QCoef_ =loadCoeff; };

    /// Returns the solution
    ParGridFunction& GetSolution(){return solgf;}

    /// Returns the solution vector.
    Vector& GetSolutionVec(){return solgf;}
    Vector GetSolutionTVec(){
      solgf.SetTrueVector();
      return solgf.GetTrueVector();}

    /// Returns the adjoint solution vector.
    // Vector& GetAdj(){return adj;}

    ParLinearForm * GetImplicitDqDx(){ return dQdx_; };
    Vector GetImplicitDqDxVec(){ return *dQdx_; };

    ParLinearForm * GetImplicitDqDxshape(){ return dQdxshape_; };

    ParLinearForm * GetImplicitDqDu(){ return dQdu_; };

private:
    ParMesh* pmesh;

    Vector X0_;

    //solution true vector
    // Vector sol;
    // Vector adj;
    Vector rhs;
    ParGridFunction solgf;
    // ParGridFunction adjgf;
    ParGridFunction bcGridFunc_;

    ParLinearForm * dQdx_;
    ParLinearForm * dQdxshape_;
    ParLinearForm * dQdu_;

    FiniteElementCollection *fec;
    ParFiniteElementSpace	  *temp_fes_;
    ParFiniteElementSpace	  *coord_fes_;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    // holds NBC in coefficient form
    std::map<int, Coefficient*> ncc;

    Array<int> ess_tdof_list_;

    ParGridFunction* QGF_ = nullptr;
    VectorCoefficient * QCoef_ = nullptr;

    Coefficient * radius_;
    ProductCoefficient *pradius_ = nullptr;

    bool GFSet = false;
    bool coeffSet = false;
};
}
#endif
