#ifndef TMOP_AD_ERR_HPP
#define TMOP_AD_ERR_HPP

#include "mfem.hpp"

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

class NodeAwareTMOPQuality
{
public:
    NodeAwareTMOPQuality(mfem::ParMesh* mesh_, int order_, TMOP_QualityMetric *metric, TargetConstructor *target_c)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        fec = new H1_FECollection(order_,dim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        X0_.SetSpace(coord_fes_);
        designVar.SetSpace(coord_fes_);
        mfem::Vector tempX0_;
        pmesh->GetNodes(tempX0_);

        X0_ = tempX0_;

        dQdx_ = new mfem::ParLinearForm(coord_fes_);
        metric_in = metric;
        target_in = target_c;
    }

    ~NodeAwareTMOPQuality()
    {
    }

    void UpdateMesh(mfem::Vector const &U);
    double EvalQoI();
    void EvalQoIGrad();

    mfem::ParLinearForm * GetDQDx(){ return dQdx_; };

    void SetDesign( mfem::ParGridFunction & design){ designVar = design; };

    private:

    mfem::ParMesh* pmesh;
    mfem::ParGridFunction X0_;
    mfem::ParGridFunction designVar;

    mfem::FiniteElementCollection *fec;
    mfem::ParFiniteElementSpace	  *coord_fes_;

    mfem::ParLinearForm * dQdx_;
    TMOP_QualityMetric *metric_in = nullptr;
    TargetConstructor *target_in = nullptr;
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




class Energy_QoI : public QoIBaseCoefficient {
public:
  Energy_QoI(mfem::ParGridFunction * solutionField, mfem::Coefficient * force, VectorCoefficient * forceGrad, int Dim)
    : solutionField_(solutionField), force_(force), forceGrad_(forceGrad), Dim_(Dim)
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

  const Vector DerivativeExactWRTX(ElementTransformation &T, const IntegrationPoint &ip) override
  {
    Vector trueGrad;
    forceGrad_->Eval(trueGrad, T, ip);

    double fieldVal = solutionField_->GetValue( T, ip );
    trueGrad *= fieldVal;
    return trueGrad;
  }

private:

  mfem::ParGridFunction * solutionField_;
  mfem::Coefficient * force_;
  VectorCoefficient * forceGrad_ = nullptr;

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
    QuantityOfInterest(ParMesh* mesh_, enum QoIType qoiType, int order_, int physics_order_, Array<int> NeumannBdr = {} ,int pdim = 1)
    : pmesh(mesh_), qoiType_(qoiType), bdr(NeumannBdr)
    {
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        pfec = new H1_FECollection(physics_order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,pfec,pdim);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);
        temp_fes_grad_ = new ParFiniteElementSpace(pmesh,pfec,dim*pdim);

        solgf_.SetSpace(temp_fes_);

        dQdu_ = new ParLinearForm(temp_fes_);
        dQdx_ = new ParLinearForm(coord_fes_);
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
    void SetManufacturedSolution( Coefficient * QCoef ){ QCoef_ = QCoef; }
    void SetManufacturedSolutionGrad( VectorCoefficient * QCoefGrad ){ QCoefGrad_ = QCoefGrad; }
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
    Coefficient       * QCoef_ = nullptr;
    VectorCoefficient * QCoefGrad_ = nullptr;

    ParMesh* pmesh;
    enum QoIType qoiType_;

    Vector X0_;
    Vector designVar;

    FiniteElementCollection *fec;
    FiniteElementCollection *pfec;
    ParFiniteElementSpace	  *temp_fes_;
    ParFiniteElementSpace	  *coord_fes_;
    ParFiniteElementSpace	  *temp_fes_grad_;

    ParLinearForm * dQdu_;
    ParLinearForm * dQdx_;

    ParGridFunction solgf_;

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
    PhysicsSolverBase( ParMesh* mesh_, int order_, int physics_order_)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        pfec = new H1_FECollection(physics_order_,dim);
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
    FiniteElementCollection *pfec;
    ParFiniteElementSpace	  *physics_fes_;
    ParFiniteElementSpace	  *coord_fes_;

    //solution true vector
    Vector sol;
    Vector adj;
    Vector rhs;
    ParGridFunction solgf, projsolgf;
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

class Elasticity_Solver : public PhysicsSolverBase
{
public:
    Elasticity_Solver(ParMesh* mesh_, std::vector<std::pair<int, double>> ess_bdr, const Array<int> & neumannBdr, int order_)
    : PhysicsSolverBase( mesh_, order_ , order_), bdr(neumannBdr)
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

        firstLameCoef = new ConstantCoefficient(0.5769230769);
        secondLameCoef = new ConstantCoefficient(1.0/2.6);
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

    void setMaterial( Coefficient * firstLameCoef_, Coefficient * secondLameCoef_)
    {
      // delete(firstLameCoef);
      // delete(secondLameCoef);
      firstLameCoef = firstLameCoef_;
      secondLameCoef = secondLameCoef_;
    }

private:

    // holds NBC in coefficient form
    std::map<int, Coefficient*> ncc;

    Array<int> ess_tdof_list_;

    VectorCoefficient * QCoef_ = nullptr;

    Array<int> bdr;

    Coefficient * firstLameCoef = nullptr;
    Coefficient * secondLameCoef = nullptr;
};

class VectorHelmholtz
{
public:
    VectorHelmholtz(ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, real_t radius, int order_, int physics_order_)
    {

        radius_ = new ConstantCoefficient(radius);
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        pfec = new H1_FECollection(physics_order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,pfec,dim);
        temp_fes_scalar_ = new ParFiniteElementSpace(pmesh,pfec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        // sol.SetSize(coord_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(coord_fes_->GetTrueVSize()); rhs=0.0;
        // adj.SetSize(coord_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(temp_fes_);
        // adjgf.SetSpace(coord_fes_);

        dQdx_ = new ParLinearForm(coord_fes_);
        dQdu_ = new ParLinearForm(temp_fes_scalar_);
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

    VectorHelmholtz(ParMesh* mesh_, std::vector<std::pair<int, int>> ess_bdr, ProductCoefficient *radius, int order_, int physics_order_)
    {
        pradius_ = radius;
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        pmesh->GetNodes(X0_);

        fec = new H1_FECollection(order_,dim);
        pfec = new H1_FECollection(physics_order_,dim);
        temp_fes_ = new ParFiniteElementSpace(pmesh,pfec, dim);
        temp_fes_scalar_ = new ParFiniteElementSpace(pmesh,pfec);
        coord_fes_ = new ParFiniteElementSpace(pmesh,fec,dim);

        // sol.SetSize(coord_fes_->GetTrueVSize()); sol=0.0;
        rhs.SetSize(coord_fes_->GetTrueVSize()); rhs=0.0;
        // adj.SetSize(coord_fes_->GetTrueVSize()); adj=0.0;

        solgf.SetSpace(temp_fes_);
        // adjgf.SetSpace(coord_fes_);

        dQdx_ = new ParLinearForm(coord_fes_);
        dQdu_ = new ParLinearForm(temp_fes_scalar_);
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
    FiniteElementCollection *pfec;
    ParFiniteElementSpace	  *temp_fes_;
    ParFiniteElementSpace	  *coord_fes_;
    ParFiniteElementSpace	  *temp_fes_scalar_;

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

class DiffusionSolver
{
private:
   mfem::Mesh * mesh = nullptr;
   int order = 1;
   // diffusion coefficient
   mfem::Coefficient * diffcf = nullptr;
   // mass coefficient
   mfem::Coefficient * masscf = nullptr;
   mfem::Coefficient * rhscf = nullptr;
   mfem::Coefficient * essbdr_cf = nullptr;
   mfem::Coefficient * neumann_cf = nullptr;
   mfem::VectorCoefficient * gradient_cf = nullptr;

   // FEM solver
   int dim;
   mfem::FiniteElementCollection * fec = nullptr;
   mfem::FiniteElementSpace * fes = nullptr;
   mfem::Array<int> ess_bdr;
   mfem::Array<int> neumann_bdr;
   mfem::GridFunction * u = nullptr;
   mfem::LinearForm * b = nullptr;
   bool parallel;
#ifdef MFEM_USE_MPI
   mfem::ParMesh * pmesh = nullptr;
   mfem::ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   DiffusionSolver() { }
   DiffusionSolver(mfem::Mesh * mesh_, int order_, mfem::Coefficient * diffcf_,
                   mfem::Coefficient * cf_);

   void SetMesh(mfem::Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<mfem::ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetOrder(int order_) { order = order_ ; }
   void SetDiffusionCoefficient(mfem::Coefficient * diffcf_) { diffcf = diffcf_; }
   void SetMassCoefficient(mfem::Coefficient * masscf_) { masscf = masscf_; }
   void SetRHSCoefficient(mfem::Coefficient * rhscf_) { rhscf = rhscf_; }
   void SetEssentialBoundary(const mfem::Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const mfem::Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetNeumannData(mfem::Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
   void SetEssBdrData(mfem::Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
   void SetGradientData(mfem::VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

   void ResetFEM();
   void SetupFEM();

   void Solve();
   mfem::GridFunction * GetFEMSolution();
   mfem::LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   mfem::ParGridFunction * GetParFEMSolution();
   mfem::ParLinearForm * GetParLinearForm()
   {
      if (parallel)
      {
         return dynamic_cast<mfem::ParLinearForm *>(b);
      }
      else
      {
         MFEM_ABORT("Wrong code path. Call GetLinearForm");
         return nullptr;
      }
   }
#endif

   ~DiffusionSolver();

};

}
#endif
