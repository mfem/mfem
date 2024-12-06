#include "tmop_ad_err.hpp"

namespace mfem {

void IdentityMatrix(int dim, DenseMatrix &I)
{
  I.SetSize(dim, dim);
  I = 0.0;
  for (int i = 0; i < dim; i++) {
    I(i, i) = 1.0;
  }
}

void Vectorize(const DenseMatrix &A, Vector &a)
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

double MatrixInnerProduct(const DenseMatrix &A, const DenseMatrix &B)
{
  double inner_product = 0.0;
  for (int i = 0; i < A.NumRows(); i++) {
    for (int j = 0; j < A.NumCols(); j++) {
      inner_product += A(i, j) * B(i, j);
    }
  }
  return inner_product;
}

void ConjugationProduct(const DenseMatrix &A, const DenseMatrix &B, const DenseMatrix &C, DenseMatrix &D)
{
  DenseMatrix CBt(C.NumRows(), B.NumRows());
  MultABt(C, B, CBt);
  Mult(A, CBt, D);
}

LFNodeCoordinateSensitivityIntegrator::LFNodeCoordinateSensitivityIntegrator( int IntegrationOrder)
  : IntegrationOrder_(IntegrationOrder)
{}

void LFNodeCoordinateSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  DenseMatrix dN(dof, dim);
  DenseMatrix NxPhix(dof, dim);
  DenseMatrix B(dof, dim);
  DenseMatrix IxB(dof, dim);
  Vector IxBTvec(dof * dim);
  Vector IxN_vec(dof * dim);
  Vector N(dof);
  DenseMatrix graduDerivxB(dof, dim);
  Vector graduDerivxBvec(dof * dim);

  // output vector
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), integrationOrder);

  // identity tensor
  DenseMatrix I;
  IdentityMatrix(dim, I);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    // evaluate shape function derivative
    el.CalcDShape(ip, dN);
    el.CalcShape(ip, N);
    DenseMatrix matN(N.GetData(), dof, 1);

    // get inverse jacobian
    DenseMatrix Jinv = T.InverseJacobian();
    Mult(dN, Jinv, B);

    // term 1
    Mult(B, QoI_->gradTimesexplicitSolutionGradientDerivative(T, ip), graduDerivxB);
    Vectorize(graduDerivxB, graduDerivxBvec);
    elvect.Add( -1.0 * w , graduDerivxBvec);

    // term 2
    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * QoI_->Eval(T, ip), IxBTvec);

    // term 3
    Mult(matN, QoI_->explicitShapeDerivative(T, ip), NxPhix);
    Vectorize(NxPhix, IxN_vec);
    elvect.Add(w , IxN_vec);
  }
}

LFAvgErrorNodeCoordinateSensitivityIntegrator::LFAvgErrorNodeCoordinateSensitivityIntegrator(
  mfem::ParGridFunction * solutionField, GridFunctionCoefficient * elementVol, int IntegrationOrder)
  : solutionField_(solutionField), elementVol_(elementVol), IntegrationOrder_(IntegrationOrder)
{}

void LFAvgErrorNodeCoordinateSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
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
  mfem::DenseMatrix graduDerivxB(dof, dim);
  mfem::Vector graduDerivxBvec(dof * dim);
  mfem::Vector shapeSum(dim * dof); shapeSum = 0.0;

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

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
      const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);

      double solVal = solutionField_->GetValue( T, ip );
      solVal = solVal-1.0;

      el.CalcDShape(ip, dN);

      // get inverse jacobian
      mfem::DenseMatrix Jinv = T.InverseJacobian();
      mfem::Mult(dN, Jinv, B);

      double vol = elementVol_->Eval( T, ip );

      // evaluate gaussian integration weight
      double w = ip.weight * T.Weight();
      mfem::Mult(B, I, IxB);
      Vectorize(IxB, IxBTvec);      
      shapeSum.Add( w * solVal / (vol*vol), IxBTvec);
  }

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    // evaluate shape function derivative
    el.CalcDShape(ip, dN);

    // get inverse jacobian
    mfem::DenseMatrix Jinv = T.InverseJacobian();
    mfem::Mult(dN, Jinv, B);

    // term 1
    // mfem::Mult(B, QoI_->gradTimesexplicitSolutionGradientDerivative(el,T, ip), graduDerivxB);
    // Vectorize(graduDerivxB, graduDerivxBvec);
    // elvect.Add( -1.0 * w , graduDerivxBvec);

    // term 2
    mfem::Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * QoI_->Eval(T, ip), IxBTvec);

    const mfem::DenseMatrix & derivVal = QoI_->explicitSolutionDerivative(T, ip);
    double val = derivVal.Elem(0,0);

    elvect.Add( -1.0*w *val, shapeSum);

    // term 3
    // Mult(matN, QoI_->explicitShapeDerivative(el,T, ip), NxPhix);
    // Vectorize(NxPhix, IxN_vec);
    // elvect.Add(w , IxN_vec);
  }
}

LFErrorIntegrator::LFErrorIntegrator( int IntegrationOrder)
  : IntegrationOrder_(IntegrationOrder)
{}

void LFErrorIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  //int dim = el.GetDim();

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), integrationOrder);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);
    elvect.Add(w  * QoI_->Eval(T, ip), N);
  }
}

LFErrorDerivativeIntegrator::LFErrorDerivativeIntegrator( int IntegrationOrder)
  : IntegrationOrder_(IntegrationOrder)
{}

void LFErrorDerivativeIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  Vector N(dof);
  DenseMatrix dN(dof, dim);
  DenseMatrix B(dof, dim);
  DenseMatrix BT(dim, dof);
  Vector DQdgradxdNdxvec(dof);
  DenseMatrix DQdgradxdNdx(1, dof);

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), integrationOrder);

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcDShape(ip, dN);
    el.CalcShape(ip, N);
    // get inverse jacobian
    DenseMatrix Jinv = T.InverseJacobian();
    Mult(dN, Jinv, B);

    // term 1
    const DenseMatrix & SolGradDeriv = QoI_->explicitSolutionGradientDerivative(T, ip);

    BT.Transpose(B);

    Mult(SolGradDeriv, BT, DQdgradxdNdx);
    Vectorize(DQdgradxdNdx, DQdgradxdNdxvec);
    elvect.Add(w , DQdgradxdNdxvec);

    //term 2
    const DenseMatrix & derivVal = QoI_->explicitSolutionDerivative(T, ip);
    double val = derivVal.Elem(0,0);

    elvect.Add(w  * val, N);
  }
}

LFErrorDerivativeIntegrator_2::LFErrorDerivativeIntegrator_2( ParFiniteElementSpace * fespace, Array<int> count,int IntegrationOrder)
  : fespace_(fespace), count_(count), IntegrationOrder_(IntegrationOrder)
{}

void LFErrorDerivativeIntegrator_2::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), integrationOrder);

  // initialize storage
  Vector N(dof);
  Vector shape(dof);
  DenseMatrix dN(dof, dim);
  DenseMatrix B(dof, dim);
  DenseMatrix BT(dim, dof);
  Vector DQdgradxdNdxvec(dof);
  DenseMatrix DQdgradxdNdx(1, dof);

  DenseMatrix SumdNdxatNodes(dof, dof*dim); SumdNdxatNodes = 0.0;

  //-----------------------------------------------------------------------------------

  const IntegrationRule *ir_p = &el.GetNodes();
  int fnd = ir_p->GetNPoints();
  //flux.SetSize( fnd * spaceDim );

  DenseMatrix dshape(dof,dim), invdfdx(dim, dim);

  int EleNo = T.ElementNo;

  Array<int> fdofs;
  fespace_->GetElementVDofs(EleNo, fdofs);

  for (int i = 0; i < fnd; i++)
  {
      const IntegrationPoint &ip = ir_p->IntPoint(i);
      T.SetIntPoint (&ip);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);
      // dshape.MultTranspose(u, vec);

      CalcInverse(T.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, B);
      //invdfdx.MultTranspose(vec, vecdxt);

      std::cout<<"dshape H | W: "<<dshape.Height()<<" | "<<dshape.Width()<<std::endl;
      std::cout<<"B H | W: "<<B.Height()<<" | "<<B.Width()<<std::endl;
      std::cout<<"shape S: "<<shape.Size()<< std::endl;
      std::cout<<"count: "<<count_[fdofs[i]]<< std::endl;

    //  B.Print();
    //  SumdNdxatNodes.Print();

      for( int jj = 0; jj < dof; jj++)
      {

        // deveide here by node weights  // FIXME
        SumdNdxatNodes(jj,i) = B(jj,0) / count_[fdofs[i]];
        SumdNdxatNodes(jj,dof+i) = B(jj,1) / count_[fdofs[i]];

        // SumdNdxatNodes(jj,i) = B(jj,0) / 4.0;
        // SumdNdxatNodes(jj,dof+i) = B(jj,1) / 4.0;
      }

      // shape.Print();

  }

  // SumdNdxatNodes.Print();

   //mfem_error("stop in LFErrorDerivativeIntegrator_2");





  //-----------------------------------------------------------------------------------



  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcDShape(ip, dN);
    el.CalcShape(ip, N);
    // get inverse jacobian
    DenseMatrix Jinv = T.InverseJacobian();
    Mult(dN, Jinv, B);

    // term 1
    const DenseMatrix & SolGradDeriv = QoI_->explicitSolutionGradientDerivative(T, ip);

    BT.Transpose(B);

    Mult(SolGradDeriv, BT, DQdgradxdNdx);
    Vectorize(DQdgradxdNdx, DQdgradxdNdxvec);
    elvect.Add(w , DQdgradxdNdxvec);

    //term 2
    DenseMatrix Nhelper(dof*dim, dim); Nhelper = 0.0;
    DenseMatrix NtimesSumdNdx(dof, dim); NtimesSumdNdx = 0.0;
    DenseMatrix NtimesSumdNdxT(dim, dof); NtimesSumdNdxT = 0.0;

      for( int jj = 0; jj < dof; jj++)
      {
        Nhelper(jj,0) =     N(jj);
        Nhelper(dof+jj,1) = N(jj);
      }

    Mult(SumdNdxatNodes, Nhelper, NtimesSumdNdx);


    NtimesSumdNdxT.Transpose(NtimesSumdNdx);
    Mult(SolGradDeriv, NtimesSumdNdxT, DQdgradxdNdx);
    Vectorize(DQdgradxdNdx, DQdgradxdNdxvec);
    elvect.Add(-1.0*w , DQdgradxdNdxvec);



    //term 2
    // const DenseMatrix & derivVal = QoI_->explicitSolutionDerivative(T, ip);
    // double val = derivVal.Elem(0,0);

    // elvect.Add(w  * val, N);
  }
}

LFAverageErrorDerivativeIntegrator::LFAverageErrorDerivativeIntegrator( ParFiniteElementSpace * fespace, GridFunctionCoefficient * elementVol,int IntegrationOrder)
  : fespace_(fespace), elementVol_(elementVol), IntegrationOrder_(IntegrationOrder)
{}

void LFAverageErrorDerivativeIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
    mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  int integrationOrder = 2 * el.GetOrder() + 2;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), integrationOrder);

  // initialize storage
  mfem::Vector N(dof);
  mfem::Vector shape(dof);
  mfem::Vector shapeSum(dof); shapeSum = 0.0;

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
      const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);

      // evaluate gaussian integration weight
      double w = ip.weight * T.Weight();
      el.CalcShape(ip, shape);

      double vol = elementVol_->Eval( T, ip );

      shapeSum.Add(w*vol, shape);
  }

  //-----------------------------------------------------------------------------------

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    // term 2
    const mfem::DenseMatrix & derivVal = QoI_->explicitSolutionDerivative( T, ip);
    double val = derivVal.Elem(0,0);

    elvect.Add(w * val, N);
    elvect.Add(-1.0*w * val, shapeSum);
  }
}

ThermalConductivityShapeSensitivityIntegrator::ThermalConductivityShapeSensitivityIntegrator(
  Coefficient &conductivity, const ParGridFunction &t_primal, const ParGridFunction &t_adjoint)
  : k_(&conductivity), t_primal_(&t_primal), t_adjoint_(&t_adjoint)
{}

void ThermalConductivityShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  DenseMatrix dN(dof, dim);
  DenseMatrix B(dof, dim);

  DenseMatrix BBT(dof, dof);
  DenseMatrix dBBT(dof, dof);
  DenseMatrix BdBT(dof, dof);

  DenseMatrix dX_dXk(dim, dof);
  DenseMatrix dJ_dXk(dim, dim);
  DenseMatrix dJinv_dXk(dim, dim);
  DenseMatrix dB_dXk(dof, dim);

  DenseMatrix dK_dXk(dof, dof);

  Vector te_primal(dof);
  Vector te_adjoint(dof);

  Array<int> vdofs;
  t_primal_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_primal_->GetSubVector(vdofs, te_primal);

  t_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_adjoint_->GetSubVector(vdofs, te_adjoint);

  // output matrix
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));

  // loop over nodal coordinates (X_k)
  for (int m = 0; m < dim; m++) {
    for (int n = 0; n < dof; n++) {
      dX_dXk = 0.0;
      dX_dXk(m, n) = 1.0;

      dK_dXk = 0.0;
      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++) {
        const IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);

        double w = ip.weight * T.Weight();
        el.CalcDShape(ip, dN);
        Mult(dN, T.InverseJacobian(), B);

        // compute derivative of Jacobian w.r.t. nodal coordinate
        Mult(dX_dXk, dN, dJ_dXk);

        // compute derivative of J^(-1)
        DenseMatrix JinvT = T.InverseJacobian();
        JinvT.Transpose();
        ConjugationProduct(T.InverseJacobian(), JinvT, dJ_dXk, dJinv_dXk);
        dJinv_dXk *= -1.0;

        // compute derivative of B w.r.t. nodal coordinate
        Mult(dN, dJinv_dXk, dB_dXk);

        // compute derivative of stiffness matrix w.r.t. X_k
        MultAAt(B, BBT);
        MultABt(dB_dXk, B, dBBT);
        MultABt(B, dB_dXk, BdBT);

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

ThermalHeatSourceShapeSensitivityIntegrator::ThermalHeatSourceShapeSensitivityIntegrator(Coefficient &heatSource,
    const ParGridFunction &t_adjoint,
    int oa, int ob)
  : Q_(&heatSource), t_adjoint_(&t_adjoint), oa_(oa), ob_(ob)
{}

void ThermalHeatSourceShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  DenseMatrix dX_dXk(dim, dof);
  DenseMatrix dJ_dXk(dim, dim);
  Vector N(dof);
  DenseMatrix dN(dof, dim);
  Vector dp_dXk(dof);

  Vector te_adjoint(dof);

  Array<int> vdofs;
  t_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
  t_adjoint_->GetSubVector(vdofs, te_adjoint);

  // output vector
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), oa_ * el.GetOrder() + ob_);

  // loop over nodal coordinates (X_k)
  for (int m = 0; m < dim; m++) {
    for (int n = 0; n < dof; n++) {
      dX_dXk = 0.0;
      dX_dXk(m, n) = 1.0;

      dp_dXk = 0.0;
      // loop over integration points
      for (int i = 0; i < ir->GetNPoints(); i++) {
        // set integration point
        const IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);
        double w = ip.weight * T.Weight();

        // evaluate shape functions and their derivatives
        el.CalcShape(ip, N);
        el.CalcDShape(ip, dN);

        // compute derivative of Jacobian w.r.t. nodal coordinate
        Mult(dX_dXk, dN, dJ_dXk);

        // compute derivative of J^(-1)
        DenseMatrix JinvT = T.InverseJacobian();
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

void QuantityOfInterest::UpdateMesh(Vector const &U)
{
  Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

void Diffusion_Solver::UpdateMesh(Vector const &U)
{
  Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

double QuantityOfInterest::EvalQoI()
{
  this->UpdateMesh(designVar);

  // make \nabla T vector coefficient

  ParGridFunction oneGridFunction = ParGridFunction(temp_fes_);
  oneGridFunction = 1.0;
  ConstantCoefficient one(1.0);
  BilinearFormIntegrator *integ = nullptr;
  ParGridFunction flux(coord_fes_);

  //------------------------------------------

  mfem::L2_FECollection fecGF_0(0, pmesh->Dimension(), mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fesGF_0(pmesh, &fecGF_0, 1);
  mfem::GridFunctionCoefficient tL2Coeff(&solgf_);
  mfem::ConstantCoefficient tConstCoeff(1.0);

  mfem::ParLinearForm b(&fesGF_0);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(tL2Coeff, 4, 6));
  b.Assemble();
  mfem::ParLinearForm b_Vol(&fesGF_0);
  b_Vol.AddDomainIntegrator(new mfem::DomainLFIntegrator(tConstCoeff, 4, 6));
  b_Vol.Assemble();

  for(uint Ik =0; Ik<b_Vol.Size(); Ik++)
  {
    b[Ik] = b[Ik] / b_Vol[Ik];
  }

  mfem::HypreParVector* assembledVector(b.ParallelAssemble());
  mfem::ParGridFunction b_GF(b.ParFESpace(), assembledVector);
  GridFunctionCoefficient L2Field(&b_GF);

  switch (qoiType_) {
  case 0:
     if( trueSolution_ == nullptr ){ mfem_error("true solution not set.");}
     ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_);
     break;
  case 1:
    if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution not set.");}
    ErrorCoefficient_ = std::make_shared<H1Error_QoI>(&solgf_, trueSolutionGrad_);
    break;
  case 2:
    integ = new DiffusionIntegrator(one);
    solgf_.ComputeFlux(*integ, flux, false);
    trueSolutionGrad_ = new VectorGridFunctionCoefficient( &flux);
    ErrorCoefficient_ = std::make_shared<ZZError_QoI>(&solgf_, trueSolutionGrad_);
    break;
  case 3:
    ErrorCoefficient_ = std::make_shared<AvgError_QoI>(&solgf_, &L2Field);
    break;
  default:
    std::cout << "Unknown Error Coeff: " << qoiType_ << std::endl;
  }

  ParGridFunction ErrorGF = ParGridFunction(temp_fes_);

  ParLinearForm scalarErrorForm(temp_fes_);
  LFErrorIntegrator *lfi = new LFErrorIntegrator;
  lfi->SetQoI(ErrorCoefficient_);
  lfi->SetIntRule(&IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
  scalarErrorForm.AddDomainIntegrator(lfi);
  scalarErrorForm.Assemble();

  return scalarErrorForm(oneGridFunction);
}

void QuantityOfInterest::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ConstantCoefficient one(1.0);
  BilinearFormIntegrator *integ = nullptr;
  ParGridFunction flux(coord_fes_);

     //------------------------------------------

  mfem::L2_FECollection fecGF_0(0, pmesh->Dimension(), mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fesGF_0(pmesh, &fecGF_0, 1);
  mfem::GridFunctionCoefficient tL2Coeff(&solgf_);
  mfem::ConstantCoefficient tConstCoeff(1.0);

  mfem::ParLinearForm b(&fesGF_0);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(tL2Coeff, 4, 6));
  b.Assemble();
  mfem::ParLinearForm b_Vol(&fesGF_0);
  b_Vol.AddDomainIntegrator(new mfem::DomainLFIntegrator(tConstCoeff, 4, 6));
  b_Vol.Assemble();

  for(uint Ik =0; Ik<b_Vol.Size(); Ik++)
  {
    b[Ik] = b[Ik] / b_Vol[Ik];
  }

  mfem::HypreParVector* assembledVector(b.ParallelAssemble());
  mfem::ParGridFunction b_GF(b.ParFESpace(), assembledVector);
  mfem::HypreParVector* assembledVector_2(b_Vol.ParallelAssemble());
  mfem::ParGridFunction bVol_GF(b.ParFESpace(), assembledVector_2);
  GridFunctionCoefficient L2Field(&b_GF);
  GridFunctionCoefficient L2VolField(&bVol_GF);

  switch (qoiType_) {
    case 0:
      if( trueSolution_ == nullptr ){ mfem_error("true solution not set.");}
      ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_);
      break;
    case 1:
      if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution not set.");}
      ErrorCoefficient_ = std::make_shared<H1Error_QoI>(&solgf_, trueSolutionGrad_);
      break;
    case 2:
      integ = new DiffusionIntegrator(one);
      solgf_.ComputeFlux(*integ, flux, false);
      trueSolutionGrad_ = new VectorGridFunctionCoefficient( &flux);
      ErrorCoefficient_ = std::make_shared<ZZError_QoI>(&solgf_, trueSolutionGrad_);
      break;
    case 3:
      ErrorCoefficient_ = std::make_shared<AvgError_QoI>(&solgf_, &L2Field);
      break;
    default:
      std::cout << "Unknown Error Coeff: " << qoiType_ << std::endl;
   }

  if(qoiType_ == QoIType::ZZ_ERROR)
  {
    // evaluate grad wrt temp
    {
        int nfe = temp_fes_->GetNE();


        Array<int> fdofs;
        DofTransformation *fdoftrans;
        Array<int> count(solgf_.Size()); count = 0;

        for (int i = 0; i < nfe; i++)
        {
          fdoftrans = temp_fes_->GetElementVDofs(i, fdofs);
          FiniteElementSpace::AdjustVDofs(fdofs);

          for (int j = 0; j < fdofs.Size(); j++)
          {
            count[fdofs[j]]++;
          }
        }

      ParLinearForm T_gradForm(temp_fes_);
      LFErrorDerivativeIntegrator_2 *lfi = new LFErrorDerivativeIntegrator_2(temp_fes_, count);
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add( 1.0, T_gradForm);

      //solgf_.ComputeFlux(*integ, flux, false);
    }
      // evaluate grad wrt coord
    {
      LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator;
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&IntRules.Get(coord_fes_->GetFE(0)->GetGeomType(), 8));

      ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddDomainIntegrator(lfi);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);
    }

  }
  else if(qoiType_ == QoIType::AVG_ERROR)
  {
    {
      ::mfem::ParLinearForm T_gradForm(temp_fes_);
      LFAverageErrorDerivativeIntegrator *lfi = new LFAverageErrorDerivativeIntegrator(temp_fes_, &L2VolField);
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&mfem::IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add( 1.0, T_gradForm);
    }
      // evaluate grad wrt coord
    {
      LFAvgErrorNodeCoordinateSensitivityIntegrator *lfi = new LFAvgErrorNodeCoordinateSensitivityIntegrator(&solgf_, &L2VolField);
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&mfem::IntRules.Get(coord_fes_->GetFE(0)->GetGeomType(), 8));

      ::mfem::ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddDomainIntegrator(lfi);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);
    }
  }
  else
  {
    // evaluate grad wrt temp
    {
      ParLinearForm T_gradForm(temp_fes_);
      LFErrorDerivativeIntegrator *lfi = new LFErrorDerivativeIntegrator;
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add( 1.0, T_gradForm);
    }

    // evaluate grad wrt coord
    {
      LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator;
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&IntRules.Get(coord_fes_->GetFE(0)->GetGeomType(), 8));

      ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddDomainIntegrator(lfi);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);
    }
  }
}

void Diffusion_Solver::FSolve()
{
  this->UpdateMesh(designVar);

  Array<int> ess_tdof_list(ess_tdof_list_);

  // assemble LHS matrix
  ConstantCoefficient kCoef(1.0);

  ParBilinearForm kForm(temp_fes_);
  ParLinearForm QForm(temp_fes_);
  kForm.AddDomainIntegrator(new DiffusionIntegrator(kCoef));

  QForm.AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));

  kForm.Assemble();
  QForm.Assemble();

  // solve for temperature
  ParGridFunction &T = solgf;

  HypreParMatrix A;
  Vector X, B;
  kForm.FormLinearSystem(ess_tdof_list, T, QForm, A, X, B);

  HypreBoomerAMG amg(A);
  amg.SetPrintLevel(0);

  CGSolver cg(temp_fes_->GetParMesh()->GetComm());
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(500);
  cg.SetPreconditioner(amg);
  cg.SetOperator(A);
  cg.Mult(B, X);

  kForm.RecoverFEMSolution(X, QForm, T);
}

void Diffusion_Solver::ASolve( Vector & rhs )
{
    // the nodal coordinates will default to the initial mesh
    this->UpdateMesh(designVar);

    Array<int> ess_tdof_list(ess_tdof_list_);

    // assemble LHS matrix
    ConstantCoefficient kCoef(1.0);

    ParBilinearForm kForm(temp_fes_);
    kForm.AddDomainIntegrator(new DiffusionIntegrator(kCoef));
    kForm.Assemble();

    // solve adjoint problem
    ParGridFunction adj_sol(temp_fes_);
    adj_sol = 0.0;

    HypreParMatrix A;
    Vector X, B;
    kForm.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    HypreBoomerAMG amg(A);
    amg.SetPrintLevel(0);

    CGSolver cg(temp_fes_->GetParMesh()->GetComm());
    cg.SetRelTol(1e-10);
    cg.SetMaxIter(500);
    cg.SetPreconditioner(amg);
    cg.SetOperator(A);
    cg.Mult(B, X);

    kForm.RecoverFEMSolution(X, rhs, adj_sol);

    ParLinearForm LHS_sensitivity(coord_fes_);
    LHS_sensitivity.AddDomainIntegrator(new ThermalConductivityShapeSensitivityIntegrator(kCoef, solgf, adj_sol));
    LHS_sensitivity.Assemble();

    ParLinearForm RHS_sensitivity(coord_fes_);
    RHS_sensitivity.AddDomainIntegrator(new ThermalHeatSourceShapeSensitivityIntegrator(*QCoef_, adj_sol));
    RHS_sensitivity.Assemble();

    *dQdx_ = 0.0;
    dQdx_->Add(-1.0, LHS_sensitivity);
    dQdx_->Add( 1.0, RHS_sensitivity);
}
}
