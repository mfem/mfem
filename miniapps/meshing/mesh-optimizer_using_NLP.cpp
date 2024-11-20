#include "mesh-optimizer_using_NLP.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_PETSC
#include "petsc.h"
#endif

namespace mfem {

void IdentityMatrix(int dim, mfem::DenseMatrix &I)
{
  I.SetSize(dim, dim);
  I = 0.0;
  for (int i = 0; i < dim; i++) {
    I(i, i) = 1.0;
  }
}

void Vectorize(const mfem::DenseMatrix &A, mfem::Vector &a)
{
  int m = A.NumRows();
  int n = A.NumCols();
  a.SetSize(m * n);
  int k = 0;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      a(k++) = A(i, j);
    }
  }
}

double MatrixInnerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B)
{
  double inner_product = 0.0;
  for (int i = 0; i < A.NumRows(); i++) {
    for (int j = 0; j < A.NumCols(); j++) {
      inner_product += A(i, j) * B(i, j);
    }
  }
  return inner_product;
}

void ConjugationProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, const mfem::DenseMatrix &C, mfem::DenseMatrix &D)
{
  mfem::DenseMatrix CBt(C.NumRows(), B.NumRows());
  mfem::MultABt(C, B, CBt);
  mfem::Mult(A, CBt, D);
}

 LFNodeCoordinateSensitivityIntegrator::LFNodeCoordinateSensitivityIntegrator(mfem::Coefficient &Q, int Index1, int Index2,
    int IntegrationOrder)
  : Q_(&Q), Index1_(Index1), Index2_(Index2), IntegrationOrder_(IntegrationOrder)
{}

void LFNodeCoordinateSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
    mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  mfem::DenseMatrix dN(dof, dim);
  mfem::DenseMatrix NxPhix(dof, dim);
  mfem::DenseMatrix B(dof, dim);
  mfem::DenseMatrix IxB(dof, dim);
  mfem::Vector IxBTvec(dof * dim);
  mfem::Vector IxN_vec(dof * dim);
  mfem::Vector N(dof);

  // output vector
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), integrationOrder);

  // identity tensor
  mfem::DenseMatrix I;
  IdentityMatrix(dim, I);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    // evaluate shape function derivative
    el.CalcDShape(ip, dN);
    el.CalcShape(ip, N);
    mfem::DenseMatrix matN(N.GetData(), dof, 1);

    // get inverse jacobian
    mfem::DenseMatrix Jinv = T.InverseJacobian();
    mfem::Mult(dN, Jinv, B);

    // term 2
    mfem::Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * QoI_->Eval(T, ip) * Q_->Eval(T, ip), IxBTvec);

    // term 3
    Mult(matN, QoI_->explicitShapeDerivative(T, ip), NxPhix);
    Vectorize(NxPhix, IxN_vec);
    elvect.Add(w * Q_->Eval(T, ip), IxN_vec);
  }
}

LFErrorIntegrator::LFErrorIntegrator(mfem::Coefficient &Q, int IntegrationOrder)
  : Q_(&Q), IntegrationOrder_(IntegrationOrder)
{}

void LFErrorIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
    mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  //int dim = el.GetDim();

  // initialize storage
  mfem::Vector N(dof);

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), integrationOrder);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);
    elvect.Add(w * QoI_->Eval(T, ip) * Q_->Eval(T, ip), N);
  }
}

LFErrorDerivativeIntegrator::LFErrorDerivativeIntegrator(mfem::Coefficient &Q, int IntegrationOrder)
  : Q_(&Q), IntegrationOrder_(IntegrationOrder)
{}

void LFErrorDerivativeIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
    mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  //int dim = el.GetDim();

  // initialize storage
  mfem::Vector N(dof);

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), integrationOrder);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    const mfem::DenseMatrix & derivVal = QoI_->explicitSolutionDerivative(T, ip);
    double val = derivVal.Elem(0,0);

    el.CalcShape(ip, N);
    elvect.Add(w * QoI_->Eval(T, ip) * val, N);
  }
}

ThermalConductivityShapeSensitivityIntegrator::ThermalConductivityShapeSensitivityIntegrator(
  mfem::Coefficient &conductivity, const mfem::ParGridFunction &t_primal, const mfem::ParGridFunction &t_adjoint)
  : k_(&conductivity), t_primal_(&t_primal), t_adjoint_(&t_adjoint)
{}

void ThermalConductivityShapeSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
    mfem::ElementTransformation &T, mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  mfem::DenseMatrix dN(dof, dim);
  mfem::DenseMatrix B(dof, dim);

  mfem::DenseMatrix BBT(dof, dof);
  mfem::DenseMatrix dBBT(dof, dof);
  mfem::DenseMatrix BdBT(dof, dof);

  mfem::DenseMatrix dX_dXk(dim, dof);
  mfem::DenseMatrix dJ_dXk(dim, dim);
  mfem::DenseMatrix dJinv_dXk(dim, dim);
  mfem::DenseMatrix dB_dXk(dof, dim);

  mfem::DenseMatrix dK_dXk(dof, dof);

  mfem::Vector te_primal(dof);
  mfem::Vector te_adjoint(dof);

  mfem::Array<int> vdofs;
  t_primal_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_primal_->GetSubVector(vdofs, te_primal);

  t_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_adjoint_->GetSubVector(vdofs, te_adjoint);

  // output matrix
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));

  // loop over nodal coordinates (X_k)
  for (int m = 0; m < dim; m++) {
    for (int n = 0; n < dof; n++) {
      dX_dXk = 0.0;
      dX_dXk(m, n) = 1.0;

      dK_dXk = 0.0;
      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++) {
        const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);

        double w = ip.weight * T.Weight();
        el.CalcDShape(ip, dN);
        mfem::Mult(dN, T.InverseJacobian(), B);

        // compute derivative of Jacobian w.r.t. nodal coordinate
        mfem::Mult(dX_dXk, dN, dJ_dXk);

        // compute derivative of J^(-1)
        mfem::DenseMatrix JinvT = T.InverseJacobian();
        JinvT.Transpose();
        ConjugationProduct(T.InverseJacobian(), JinvT, dJ_dXk, dJinv_dXk);
        dJinv_dXk *= -1.0;

        // compute derivative of B w.r.t. nodal coordinate
        mfem::Mult(dN, dJinv_dXk, dB_dXk);

        // compute derivative of stiffness matrix w.r.t. X_k
        mfem::MultAAt(B, BBT);
        mfem::MultABt(dB_dXk, B, dBBT);
        mfem::MultABt(B, dB_dXk, BdBT);

        // compute derivative of integration weight w.r.t. X_k
        double dw_dXk = w * MatrixInnerProduct(JinvT, dJ_dXk);

        // put together all terms of product rule
        double k = k_->Eval(T, ip);
        dK_dXk.Add(w * k, dBBT);
        dK_dXk.Add(w * k, BdBT);
        dK_dXk.Add(dw_dXk * k, BBT);
      }
      elvect(n + m * dof) += dK_dXk.InnerProduct(te_primal, te_adjoint);
    }
  }
}

ThermalHeatSourceShapeSensitivityIntegrator::ThermalHeatSourceShapeSensitivityIntegrator(mfem::Coefficient &heatSource,
    const mfem::ParGridFunction &t_adjoint,
    int oa, int ob)
  : Q_(&heatSource), t_adjoint_(&t_adjoint), oa_(oa), ob_(ob)
{}

void ThermalHeatSourceShapeSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
    mfem::ElementTransformation &T, mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  mfem::DenseMatrix dX_dXk(dim, dof);
  mfem::DenseMatrix dJ_dXk(dim, dim);
  mfem::Vector N(dof);
  mfem::DenseMatrix dN(dof, dim);
  mfem::Vector dp_dXk(dof);

  mfem::Vector te_adjoint(dof);

  mfem::Array<int> vdofs;
  t_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_adjoint_->GetSubVector(vdofs, te_adjoint);

  // output vector
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), oa_ * el.GetOrder() + ob_);

  // loop over nodal coordinates (X_k)
  for (int m = 0; m < dim; m++) {
    for (int n = 0; n < dof; n++) {
      dX_dXk = 0.0;
      dX_dXk(m, n) = 1.0;

      dp_dXk = 0.0;
      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++) {
        // set integration point
        const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);
        double w = ip.weight * T.Weight();

        // evaluate shape functions and their derivatives
        el.CalcShape(ip, N);
        el.CalcDShape(ip, dN);

        // compute derivative of Jacobian w.r.t. nodal coordinate
        mfem::Mult(dX_dXk, dN, dJ_dXk);

        // compute derivative of J^(-1)
        mfem::DenseMatrix JinvT = T.InverseJacobian();
        JinvT.Transpose();

        // compute derivative of integration weight w.r.t. X_k
        double dw_dXk = w * MatrixInnerProduct(JinvT, dJ_dXk);

        // add integration point's derivative contribution
        dp_dXk.Add(dw_dXk * Q_->Eval(T, ip), N);
      }
      elvect(n + m * dof) += InnerProduct(te_adjoint, dp_dXk);
    }
  }
}

void QuantityOfInterest::UpdateMesh(mfem::Vector const &U)
{
  ::mfem::Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

void Diffusion_Solver::UpdateMesh(mfem::Vector const &U)
{
  ::mfem::Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

void NodeAwareTMOPQuality::UpdateMesh(mfem::Vector const &U)
{
  ::mfem::Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

double QuantityOfInterest::EvalQoI()
{
  this->UpdateMesh(designVar);

  // make \nabla T vector coefficient

  ::mfem::ParGridFunction oneGridFunction = ::mfem::ParGridFunction(temp_fes_);
  oneGridFunction = 1.0;

  ::mfem::ConstantCoefficient oneCoeff(1.0);

  std::shared_ptr<QoIBaseCoefficient> ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_);

  ::mfem::ParGridFunction ErrorGF = ::mfem::ParGridFunction(temp_fes_);
  ::mfem::ParGridFunction TGF = ::mfem::ParGridFunction(temp_fes_);
  ErrorGF.ProjectCoefficient(*ErrorCoefficient_.get());
  TGF.ProjectCoefficient(*trueSolution_);


  ::mfem::ParLinearForm scalarErrorForm(temp_fes_);
  LFErrorIntegrator *lfi = new LFErrorIntegrator(oneCoeff);
  lfi->SetQoI(ErrorCoefficient_);
  lfi->SetIntRule(&mfem::IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
  scalarErrorForm.AddDomainIntegrator(lfi);
  scalarErrorForm.Assemble();

  return scalarErrorForm(oneGridFunction);
}

void QuantityOfInterest::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ::mfem::ConstantCoefficient oneCoeff(1.0);

  std::shared_ptr<QoIBaseCoefficient> ErrorCoefficient = std::make_shared<Error_QoI>(&solgf_, trueSolution_);

  // evaluate grad wrt temp
  {
  ::mfem::ParLinearForm T_gradForm(temp_fes_);
  LFErrorDerivativeIntegrator *lfi = new LFErrorDerivativeIntegrator(oneCoeff);
  lfi->SetQoI(ErrorCoefficient);
  lfi->SetIntRule(&mfem::IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
  T_gradForm.AddDomainIntegrator(lfi);
  T_gradForm.Assemble();
  *dQdu_ = 0.0;
  dQdu_->Add( 1.0, T_gradForm);
  }

  // evaluate grad wrt coord
  {
    LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator(oneCoeff);
    lfi->SetQoI(ErrorCoefficient);
    lfi->SetIntRule(&mfem::IntRules.Get(coord_fes_->GetFE(0)->GetGeomType(), 8));

    ::mfem::ParLinearForm ud_gradForm(coord_fes_);
    ud_gradForm.AddDomainIntegrator(lfi);
    ud_gradForm.Assemble();
    *dQdx_ = 0.0;
    dQdx_->Add(1.0, ud_gradForm);
  }
}

double NodeAwareTMOPQuality::EvalQoI()
{
  this->UpdateMesh(designVar);

  ::mfem::Vector Xi = X0_;
  Xi += designVar;

  int targetId = 1;
  int metricId = 2;
  int quadOrder = 8;

  // Setup the mesh quality metric
  mfem::TMOP_QualityMetric *metric = nullptr;
  switch (metricId) {
    // T-metrics
    case 1:
      metric = new mfem::TMOP_Metric_001;
      break;
    case 2:
      metric = new mfem::TMOP_Metric_002;
      break;
    case 7:
      metric = new mfem::TMOP_Metric_007;
      break;
    case 9:
      metric = new mfem::TMOP_Metric_009;
      break;
    default:
      std::cout << "Unknown metricId_: " << metricId << std::endl;
  }

  mfem::TargetConstructor::TargetType targetT;
  mfem::TargetConstructor *targetC = nullptr;
  switch (targetId) {
    case 1:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
      break;
    case 2:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE;
      break;
    case 3:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
      break;
    default:
      std::cout << "Unknown targetId: " << targetId << "\n";
  }
  if (nullptr == targetC) {
    targetC = new mfem::TargetConstructor(targetT, MPI_COMM_WORLD);
  }
  //targetC->SetNodes(X0_);

  mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric, targetC);
  mfem::IntegrationRules *irules = nullptr;
  mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
  irules = &IntRulesLo;
  TMOPInteg->SetIntegrationRules(*irules, quadOrder);

  mfem::ParNonlinearForm a(coord_fes_);
  a.AddDomainIntegrator(TMOPInteg);

  double finalTMOPEnergy = a.GetParGridFunctionEnergy(Xi);

  return finalTMOPEnergy;
}

void NodeAwareTMOPQuality::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ::mfem::Vector Xi = X0_;
  Xi += designVar;

  int targetId = 1;
  int metricId = 2;
  int quadOrder = 8;

  // Setup the mesh quality metric
  mfem::TMOP_QualityMetric *metric = nullptr;
  switch (metricId) {
    // T-metrics
    case 1:
      metric = new mfem::TMOP_Metric_001;
      break;
    case 2:
      metric = new mfem::TMOP_Metric_002;
      break;
    case 7:
      metric = new mfem::TMOP_Metric_007;
      break;
    case 9:
      metric = new mfem::TMOP_Metric_009;
      break;
    default:
      std::cout << "Unknown metricId_: " << metricId << std::endl;
  }

  mfem::TargetConstructor::TargetType targetT;
  mfem::TargetConstructor *targetC = nullptr;
  switch (targetId) {
    case 1:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
      break;
    case 2:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE;
      break;
    case 3:
      targetT = mfem::TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
      break;
    default:
      std::cout << "Unknown targetId: " << targetId << "\n";
  }
  if (nullptr == targetC) {
    targetC = new mfem::TargetConstructor(targetT, MPI_COMM_WORLD);
  }
  //targetC->SetNodes(X0_);

  mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric, targetC);
  mfem::IntegrationRules *irules = nullptr;
  mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
  irules = &IntRulesLo;
  TMOPInteg->SetIntegrationRules(*irules, quadOrder);

  mfem::ParNonlinearForm a(coord_fes_);
  a.AddDomainIntegrator(TMOPInteg);

  *dQdx_ = 0.0;

  a.Mult (Xi, *dQdx_);
}

void Diffusion_Solver::FSolve()
{
  this->UpdateMesh(designVar);

  ::mfem::Array<int> ess_tdof_list(ess_tdof_list_);

  // assemble LHS matrix
  ::mfem::ConstantCoefficient kCoef(1.0);

  ::mfem::ParBilinearForm kForm(temp_fes_);
  ::mfem::ParLinearForm QForm(temp_fes_);
  kForm.AddDomainIntegrator(new ::mfem::DiffusionIntegrator(kCoef));

  QForm.AddDomainIntegrator(new ::mfem::DomainLFIntegrator(*QCoef_));

  kForm.Assemble();
  QForm.Assemble();

  // solve for temperature
  ::mfem::ParGridFunction &T = solgf;

  ::mfem::HypreParMatrix A;
  ::mfem::Vector X, B;
  kForm.FormLinearSystem(ess_tdof_list, T, QForm, A, X, B);

  ::mfem::HypreBoomerAMG amg(A);
  amg.SetPrintLevel(0);

  ::mfem::CGSolver cg(temp_fes_->GetParMesh()->GetComm());
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(500);
  cg.SetPreconditioner(amg);
  cg.SetOperator(A);
  cg.Mult(B, X);

  kForm.RecoverFEMSolution(X, QForm, T);
}

void Diffusion_Solver::ASolve( mfem::Vector & rhs )
{
    // the nodal coordinates will default to the initial mesh
    this->UpdateMesh(designVar);

    ::mfem::Array<int> ess_tdof_list(ess_tdof_list_);

    // assemble LHS matrix
    ::mfem::ConstantCoefficient kCoef(1.0);

    ::mfem::ParBilinearForm kForm(temp_fes_);
    kForm.AddDomainIntegrator(new ::mfem::DiffusionIntegrator(kCoef));
    kForm.Assemble();

    // solve adjoint problem
    ::mfem::ParGridFunction adj_sol(temp_fes_);
    adj_sol = 0.0;

    ::mfem::HypreParMatrix A;
    ::mfem::Vector X, B;
    kForm.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    ::mfem::HypreBoomerAMG amg(A);
    amg.SetPrintLevel(0);

    ::mfem::CGSolver cg(temp_fes_->GetParMesh()->GetComm());
    cg.SetRelTol(1e-10);
    cg.SetMaxIter(500);
    cg.SetPreconditioner(amg);
    cg.SetOperator(A);
    cg.Mult(B, X);

    kForm.RecoverFEMSolution(X, rhs, adj_sol);

    ::mfem::ParLinearForm LHS_sensitivity(coord_fes_);
    LHS_sensitivity.AddDomainIntegrator(new ThermalConductivityShapeSensitivityIntegrator(kCoef, solgf, adj_sol));
    LHS_sensitivity.Assemble();

    ::mfem::ParLinearForm RHS_sensitivity(coord_fes_);
    RHS_sensitivity.AddDomainIntegrator(new ThermalHeatSourceShapeSensitivityIntegrator(*QCoef_, adj_sol));
    RHS_sensitivity.Assemble();

    *dQdx_ = 0.0;
    dQdx_->Add(-1.0, LHS_sensitivity);
    dQdx_->Add( 1.0, RHS_sensitivity);
}
}
