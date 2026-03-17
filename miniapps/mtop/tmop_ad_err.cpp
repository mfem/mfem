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

void KroneckerProduct(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    int m = A.NumRows(), n = A.NumCols();
    int p = B.NumRows(), q = B.NumCols();
    C.SetSize(m*p, n*q);
    for (int row=0; row<C.NumRows(); row++)
        for (int col=0; col<C.NumCols(); col++)
        {
            C(row, col) = A(row/p, col/q) * B(row%p, col%q);
        }
}

void IsotropicStiffnessMatrix(int dim, double mu, double lambda, DenseMatrix &C)
{
    // set C = 2*mu*P_sym
    FourthOrderSymmetrizer(dim, C);
    C *= 2*mu;

    // compute lambda*dyadic(I, I)
    DenseMatrix I, IxI;
    Vector vecI;
    IdentityMatrix(dim, I);
    Vectorize(I, vecI);
    VectorOuterProduct(vecI, vecI, IxI);
    IxI *= lambda;

    // set C = 2*mu*P_sym + lamba*dyadic(I, I)
    C += IxI;
}

void IsotropicStiffnessMatrix3D(double E, double v, DenseMatrix &C)
{
    double     mu = E   / (2*(1+v));
    double lambda = E*v / ((1+v)*(1-2*v));
    IsotropicStiffnessMatrix(3, mu, lambda, C);
}

void VectorOuterProduct(const Vector &a, const Vector &b, DenseMatrix &C)
{
    int m = a.Size();
    int n = b.Size();
    C.SetSize(m, n);
    C = 0.0;
    for (int i=0; i<C.NumRows(); i++)
        for (int j=0; j<C.NumCols(); j++)
        {
            C(i,j) = a(i)*b(j);
        }
}

void FourthOrderSymmetrizer(int dim, DenseMatrix &S)
{
    DenseMatrix I, T;
    FourthOrderIdentity(dim, I);
    FourthOrderTranspose(dim, T);
    S  = I;
    S += T;
    S *= 0.5;
}

void FourthOrderIdentity(int dim, DenseMatrix &I4)
{
    DenseMatrix I2;
    IdentityMatrix(dim, I2);
    MatrixConjugationProduct(I2, I2, I4);
}

void FourthOrderTranspose(int dim, DenseMatrix &T)
{
    T.SetSize(dim*dim, dim*dim);
    T = 0.0;
    Vector Eij, Eji;
    DenseMatrix temp;
    for (int i=0; i<dim; i++)
    {
        for (int j=0; j<dim; j++)
        {
            UnitStrain(dim, i, j, Eij);
            UnitStrain(dim, j, i, Eji);
            VectorOuterProduct(Eij, Eji, temp);
            T += temp;
        }
    }
}

void UnitStrain(int dim, int i, int j, DenseMatrix &E)
{
    E.SetSize(dim, dim);
    E       = 0.0;
    E(i, j) = 1.0;
}

// E_ij = outer(e_i, e_j)
void UnitStrain(int dim, int i, int j, Vector &E)
{
    E.SetSize(dim*dim);
    E          = 0.0;
    E(j*dim+i) = 1.0;
}

void MatrixConjugationProduct(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    KroneckerProduct(B, A, C);
}

void NodeAwareTMOPQuality::UpdateMesh(mfem::Vector const &U)
{
  ::mfem::Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

double NodeAwareTMOPQuality::EvalQoI()
{
  this->UpdateMesh(designVar);

  ::mfem::Vector Xi(X0_.Size()); Xi = X0_;
  Xi += designVar;

  mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric_in, target_in);
  // std::cout << metric_in << " " << target_in << " k10info\n";
  // mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric, targetC);
  mfem::IntegrationRules *irules = nullptr;
  mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
  irules = &IntRulesLo;
  TMOPInteg->SetIntegrationRules(*irules, 8);

  mfem::ParNonlinearForm a(coord_fes_);
  a.AddDomainIntegrator(TMOPInteg);

  double finalTMOPEnergy = a.GetParGridFunctionEnergy(Xi);

  return finalTMOPEnergy;
}

void NodeAwareTMOPQuality::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ::mfem::ParGridFunction Xi(coord_fes_); Xi = X0_;
  Xi += designVar;

  int targetId = 1;
  int metricId = 2;
  int quadOrder = 8;

  mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric_in, target_in);
  // mfem::TMOP_Integrator *TMOPInteg = new mfem::TMOP_Integrator(metric, targetC);
  mfem::IntegrationRules *irules = nullptr;
  mfem::IntegrationRules IntRulesLo(0, mfem::Quadrature1D::GaussLobatto);
  irules = &IntRulesLo;
  TMOPInteg->SetIntegrationRules(*irules, quadOrder);

  mfem::ParNonlinearForm a(coord_fes_);
  a.AddDomainIntegrator(TMOPInteg);

  *dQdx_ = 0.0;

  Xi.SetTrueVector();
  mfem::Vector& trueXi = Xi.GetTrueVector();
  ::mfem::ParGridFunction dQdx_GF(coord_fes_);
  dQdx_GF.SetTrueVector();
  mfem::Vector& truedQdx_GF = dQdx_GF.GetTrueVector();

  a.Mult(trueXi, truedQdx_GF);

  coord_fes_->GetRestrictionTransposeOperator()->Mult(truedQdx_GF, *dQdx_ );
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

  // set integration rule
  const IntegrationRule *ir = IntRule;

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

    // term 1 - e.g. (\grad u - \grad u*)
    Mult(B, QoI_->gradTimesexplicitSolutionGradientDerivative(T, ip), graduDerivxB);
    Vectorize(graduDerivxB, graduDerivxBvec);
    elvect.Add( -1.0 * w , graduDerivxBvec);

    // term 2 - (u-u*)^2 d(detJ)/dx
    // Mult(B, I, IxB);
    // Vectorize(IxB, IxBTvec);
    // elvect.Add( w * QoI_->Eval(T, ip), IxBTvec);
    Vector Bvec(B.GetData(), dof*dim);
    elvect.Add( w * QoI_->Eval(T, ip), Bvec);

    // term 3 - this is for when QoI has x inside e.g. (u * x - u* * x)^2
    Mult(matN, QoI_->explicitShapeDerivative(T, ip), NxPhix);
    Vectorize(NxPhix, IxN_vec);
    elvect.Add(w , IxN_vec);

    // Term 4 - custom derivative 2(u-u*)(-\phi_j du*/dx_a) w_q det(J)
    Vector v = QoI_->DerivativeExactWRTX(T, ip);
    for (int d = 0; d < dim; d++)
    {
      Vector elvect_temp(elvect.GetData() + d*dof, dof);
      elvect_temp.Add(w*v(d), N);
    }
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

  // set integration rule
  const IntegrationRule *ir = IntRule;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    if (gllvec_.Size() > 0)
    {
      double val = w * std::pow(gllvec_[T.ElementNo*nqptsperel+i], 2.0);
      elvect.Add(val, N);
    }
    else {
      elvect.Add(w  * QoI_->Eval(T, ip), N);
    }
  }
}

LFErrorDerivativeIntegrator::LFErrorDerivativeIntegrator( )
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

  const IntegrationRule *ir = IntRule;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
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
    // MultABt(B, SolGradDeriv, DQdgradxdNdx);
    Vectorize(DQdgradxdNdx, DQdgradxdNdxvec);
    elvect.Add(w , DQdgradxdNdxvec);

    //term 2
    const DenseMatrix & derivVal = QoI_->explicitSolutionDerivative(T, ip);
    double val = derivVal.Elem(0,0);

    elvect.Add(w  * val, N);
  }
}

LFFilteredFieldErrorDerivativeIntegrator::LFFilteredFieldErrorDerivativeIntegrator( )
{}

void LFFilteredFieldErrorDerivativeIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  Vector N(dof);

  // output vector
  elvect.SetSize(dof*dim);
  elvect = 0.0;

  const IntegrationRule *ir = IntRule;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    // term 1
    const DenseMatrix & SolGradDeriv = QoI_->explicitSolutionGradientDerivative(T, ip);

    for (int k = 0; k < dim; k++)
    {
       for (int s = 0; s < dof; s++)
       {
          elvect(dof*k+s) += w * SolGradDeriv(0,k) * N(s);
       }
    }
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

  int integrationOrder = 2 * el.GetOrder() + 6;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  // const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), integrationOrder);
    const IntegrationRule *ir = IntRule;

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

      for( int jj = 0; jj < dof; jj++)
      {
        // deveide here by node weights  // FIXME
        SumdNdxatNodes(jj,i) = B(jj,0) / count_[fdofs[i]];
        SumdNdxatNodes(jj,dof+i) = B(jj,1) / count_[fdofs[i]];
      }

  }

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

void LFAverageErrorDerivativeIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
    Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  int integrationOrder = 2 * el.GetOrder() + 3;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const IntegrationRule *ir = IntRule;

  // initialize storage
  Vector N(dof);
  Vector shape(dof);
  Vector shapeSum(dof); shapeSum = 0.0;

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);

      // evaluate gaussian integration weight
      double w = ip.weight * T.Weight();
      el.CalcShape(ip, shape);

      double vol = elementVol_->Eval( T, ip );

      shapeSum.Add(w*vol, shape);
  }

  // /* should be this ->
  shapeSum = 0.0;
  double el_vol = 0.0;
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el_vol += ip.weight * T.Weight();
      el.CalcShape(ip, shape);
      shapeSum.Add(ip.weight * T.Weight(), shape);
  }
  shapeSum *= 1.0/el_vol;
  // */

  // -----------------------------------------------------------------------------------

  // output vector
  elvect.SetSize( dof);
  elvect = 0.0;

  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    el.CalcShape(ip, N);

    // term 2
    const DenseMatrix & derivVal = QoI_->explicitSolutionDerivative( T, ip);
    double val = derivVal.Elem(0,0);

    elvect.Add(w * val, N);
    elvect.Add(-1.0*w * val, shapeSum);
  }
}


PenaltyMassShapeSensitivityIntegrator::PenaltyMassShapeSensitivityIntegrator(
  Coefficient &penalty, const ParGridFunction &t_primal, const ParGridFunction &t_adjoint)
  : penalty_(&penalty), t_primal_(&t_primal), t_adjoint_(&t_adjoint)
{}

void PenaltyMassShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dimR = el.GetDim();
  int dimS = T.GetSpaceDim();

  // initialize storage
  DenseMatrix dN(dof, dimR);
  DenseMatrix B(dof, dimS);
  DenseMatrix IxB(dof, dimS);
  Vector IxBTvec(dof * dimS);

  DenseMatrix I;
  IdentityMatrix(dimS, I);

  //I(1,1) = 0.0;

  // output matrix
  elvect.SetSize(dof * dimS);
  elvect = 0.0;

  // set integration rule
  // const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    double w = ip.weight * T.Weight();

    //el.CalcPhysDShape(T, dN);
    el.CalcDShape(ip, dN);

    Mult(dN, T.InverseJacobian(), B);

    double val_prim = t_primal_->GetValue(T, ip);
    double val_adj = t_adjoint_->GetValue(T, ip);

    double k = penalty_->Eval(T, ip);

    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * k * val_prim * val_adj, IxBTvec);
  }
}

PenaltyShapeSensitivityIntegrator::PenaltyShapeSensitivityIntegrator(
  Coefficient &t_primal, const ParGridFunction &t_adjoint, Coefficient &t_penalty, VectorCoefficient *SolGrad, int oa, int ob)
  : t_primal_(&t_primal), t_adjoint_(&t_adjoint), t_penalty_(&t_penalty), SolGradCoeff_(SolGrad), oa_(oa), ob_(ob)
{}

void PenaltyShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dimR = el.GetDim();
  int dimS = T.GetSpaceDim();

  // initialize storage
  Vector N( dof);
  DenseMatrix dN(dof, dimR);
  DenseMatrix B(dof, dimS);
  DenseMatrix IxB(dof, dimS);
  Vector IxBTvec(dof * dimS);

  DenseMatrix I;
  IdentityMatrix(dimS, I);

  //(1,1) = 0.0;

  // output matrix
  elvect.SetSize(dof * dimS);
  elvect = 0.0;

  // set integration rule
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    double w = ip.weight * T.Weight();
    el.CalcShape(ip, N);
    el.CalcDShape(ip, dN);
    Mult(dN, T.InverseJacobian(), B);

    double val_adj = t_adjoint_->GetValue(T, ip);
    double val_sol = t_primal_->Eval(T, ip);

    double k = t_penalty_->Eval(T, ip);

    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * k * val_adj * val_sol, IxBTvec);

    // spatial derivative boundary condition
    if( SolGradCoeff_ != nullptr)
    {
      Vector solGrad(dimS);
      SolGradCoeff_->Eval(solGrad,T,ip);
      for (int d = 0; d < dimS; d++)
      {
        Vector elvect_temp(elvect.GetData() + d*dof, dof);
        elvect_temp.Add(w*k*val_adj*solGrad(d), N);
      }
    }
  }
}

GradProjectionShapeSensitivityIntegrator::GradProjectionShapeSensitivityIntegrator( const ParGridFunction &t_primal, const ParGridFunction &t_adjoint, VectorCoefficient & tempCoeff)
  : t_primal_(&t_primal), t_adjoint_(&t_adjoint), tempCoeff_(&tempCoeff)
{}

void GradProjectionShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  DenseMatrix dN(dof, dim);
  DenseMatrix B(dof, dim);
  DenseMatrix IxB(dof, dim);
  Vector IxBTvec(dof * dim);

  DenseMatrix I;
  IdentityMatrix(dim, I);

  // output matrix
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    double w = ip.weight * T.Weight();
    el.CalcDShape(ip, dN);
    Mult(dN, T.InverseJacobian(), B);

    Vector val_primal(dim);
    Vector val_adjoint(dim);
    Vector val_temp(dim);

    t_primal_->GetVectorValue(T.ElementNo, ip, val_primal);
    t_adjoint_->GetVectorValue(T.ElementNo, ip , val_adjoint);
    tempCoeff_->Eval(val_temp, T, ip);

    double val_1 = val_adjoint * val_primal;
    double val_2 = val_adjoint * val_temp;

    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);

    //term 1
    elvect.Add( w * val_1 , IxBTvec);

    //term 2
    elvect.Add(-1.0 *  w * val_2 , IxBTvec);

    //term 3
    Vectorize(B, IxBTvec);
    elvect.Add( w * val_2, IxBTvec);
  }
}

ElasticityStiffnessShapeSensitivityIntegrator::ElasticityStiffnessShapeSensitivityIntegrator(
    Coefficient &lambda, Coefficient &mu,
    const ParGridFunction &u_primal, const ParGridFunction &u_adjoint)
    : lambda_(&lambda), mu_(&mu), u_primal_(&u_primal), u_adjoint_(&u_adjoint)
{
    MFEM_ASSERT(u_primal.VectorDim() == u_adjoint.VectorDim(), "Primal and adjoint solutions are not compatible");
}

void ElasticityStiffnessShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
        ElementTransformation &T,
        Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int dim = el.GetDim();

    // intialize intermediate matrices
    DenseMatrix                  dN(dof, dim);
    DenseMatrix                   B(dof, dim);
    DenseMatrix               Kr_IB(dim*dof, dim*dim);
    DenseMatrix                matC(dim*dim);
    DenseMatrix           KrIB_matC(dim*dof, dim*dim);

    DenseMatrix              dX_dXk(dim, dof);
    DenseMatrix              dJ_dXk(dim, dim);
    DenseMatrix           dJinv_dXk(dim, dim);
    DenseMatrix              dB_dXk(dof, dim);
    DenseMatrix              Kr_IdB(dim*dof, dim*dim);
    DenseMatrix          KrIdB_matC(dim*dof, dim*dim);
    DenseMatrix     KrIdB_matC_KrIB(dim*dof, dim*dof);
    DenseMatrix     KrIB_matC_KrIdB(dim*dof, dim*dof);
    DenseMatrix      KrIB_matC_KrIB(dim*dof, dim*dof);

    DenseMatrix              dK_dXk(dim*dof, dim*dof);

    Vector                ue_primal(dim*dof);
    Vector               ue_adjoint(dim*dof);

    Array<int> vdofs;
    u_primal_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
    u_primal_->GetSubVector(vdofs, ue_primal);

    u_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
    u_adjoint_->GetSubVector(vdofs, ue_adjoint);

    // identity tensort
    DenseMatrix I;
    IdentityMatrix(dim, I);

    // output vector
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // set integration rule
     const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2*T.OrderGrad(&el));
  //IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  //const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

  //  const IntegrationRule *ir = IntRule;

    // loop over nodal coordinates (X_k)
    for (int m=0; m<dim; m++)
    {
        for (int n=0; n<dof; n++)
        {
            dX_dXk = 0.0;
            dX_dXk(m, n) = 1.0;

            dK_dXk = 0.0;
            // loop over integration points
            for (int i=0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                T.SetIntPoint(&ip);                       // set current integration point
                double w = ip.weight * T.Weight();        // evaluate gaussian integration weight
                el.CalcDShape(ip, dN);                    // evaluate shape function derivative
                Mult(dN, T.InverseJacobian(), B);   // map to iso-parametric element
                KroneckerProduct(I, B, Kr_IB);  // compute Kron(B.T, I)
                IsotropicStiffnessMatrix(dim, mu_->Eval(T, ip), lambda_->Eval(T, ip), matC);
                //std::cout<<"mu: "<<mu_->Eval(T, ip)<<" | lambda: "<<lambda_->Eval(T, ip) <<std::endl;

                // compute derivative of Jacobian w.r.t. nodal coordinate
                Mult(dX_dXk, dN, dJ_dXk);

                // compute derivative of J^(-1)
                DenseMatrix JinvT = T.InverseJacobian();
                JinvT.Transpose();
                ConjugationProduct(T.InverseJacobian(), JinvT, dJ_dXk, dJinv_dXk);
                dJinv_dXk *= -1.0;

                // compute derivative of B w.r.t. nodal coordinate
                Mult(dN, dJinv_dXk, dB_dXk);
                KroneckerProduct(I, dB_dXk, Kr_IdB);

                // compute derivative of stiffness matrix w.r.t. X_k
                Mult(Kr_IdB, matC, KrIdB_matC);
                Mult(Kr_IB, matC, KrIB_matC);
                MultABt(KrIdB_matC, Kr_IB, KrIdB_matC_KrIB);
                MultABt(KrIB_matC, Kr_IdB, KrIB_matC_KrIdB);
                MultABt(KrIB_matC, Kr_IB, KrIB_matC_KrIB);

                // compute derivative of integration weight w.r.t. X_k
                double dw_dXk = w * MatrixInnerProduct(JinvT, dJ_dXk);

                // put together all terms of product rule
                                            //std::cout<<"mu111:=================================================================== " <<dw_dXk<<std::endl;
                dK_dXk.Add(w, KrIdB_matC_KrIB);
                //KrIdB_matC_KrIB.Print();
                dK_dXk.Add(w, KrIB_matC_KrIdB);
                //KrIB_matC_KrIdB.Print();
                dK_dXk.Add(dw_dXk, KrIB_matC_KrIB);
                //KrIB_matC_KrIB.Print();
            }
                            //std::cout<<"mu111: "<<dK_dXk.InnerProduct(ue_primal, ue_adjoint) <<std::endl;
            elvect(n+m*dof) += dK_dXk.InnerProduct(ue_primal, ue_adjoint);
        }
    }
}

ElasticityTractionIntegrator::ElasticityTractionIntegrator(VectorCoefficient &f, int oa, int ob)
    : f_(&f), oa_(oa), ob_(ob)
{}

void ElasticityTractionIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &T,
        Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int vdim = f_->GetVDim();

    // initialize storage
    Vector                        N(dof);
    DenseMatrix               Kr_IN(vdim*dof, vdim);
    Vector                        f(vdim);
    Vector                  Kr_IN_f(vdim*dof);

    // identity tensor
    DenseMatrix I;
    IdentityMatrix(vdim, I);

    // output vector
    elvect.SetSize(vdim*dof);
    elvect = 0.0;

    // set integration rule
    const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), oa_*el.GetOrder()+ob_);

    // loop over integration points
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        // set integration point
        const IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);

        // compute weight
        double w = ip.weight * T.Weight();

        // evaluate shape functions
        el.CalcShape(ip, N);

        // compute kronecker
        DenseMatrix matN(N.GetData(), dof, 1);
        KroneckerProduct(I, matN, Kr_IN);

        // operator on traction vector
        f_->Eval(f, T, ip);
        Kr_IN.Mult(f, Kr_IN_f);

        // add integration point's contribution
        elvect.Add(w, Kr_IN_f);
    }
}

ElasticityTractionShapeSensitivityIntegrator::ElasticityTractionShapeSensitivityIntegrator(
    VectorCoefficient &f, const ParGridFunction &u_adjoint, int oa, int ob)
    : f_(&f), u_adjoint_(&u_adjoint), oa_(oa), ob_(ob)
{}

void ElasticityTractionShapeSensitivityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
        ElementTransformation &T,
        Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int dim = el.GetDim();
    int vdim = f_->GetVDim();

    // initialize storage
    DenseMatrix              dX_dXk(vdim, dof);
    DenseMatrix              dJ_dXk(vdim, dim);
    Vector                        N(dof);
    DenseMatrix                  dN(dof, dim);
    DenseMatrix               Kr_IN(vdim*dof, vdim);
    Vector                        f(vdim);
    Vector                  Kr_IN_f(vdim*dof);
    Vector                   dp_dXk(vdim*dof);

    Vector               ue_adjoint(vdim*dof);
    Array<int> vdofs;
    u_adjoint_->ParFESpace()->GetBdrElementVDofs(T.ElementNo, vdofs);
    u_adjoint_->GetSubVector(vdofs, ue_adjoint);

    // identity tensor
    DenseMatrix I;
    IdentityMatrix(vdim, I);

    // output vector
    elvect.SetSize(vdim*dof);
    elvect = 0.0;

    // set integration rule
    const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), oa_*el.GetOrder()+ob_);

    // loop over nodal coordinates (X_k)
    for (int m=0; m<vdim; m++)
    {
        for (int n=0; n<dof; n++)
        {
            dX_dXk = 0.0;
            dX_dXk(m, n) = 1.0;

            dp_dXk = 0.0;
            // loop over integration points
            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                // set integration point
                const IntegrationPoint &ip = ir->IntPoint(i);
                T.SetIntPoint(&ip);

                // evaluate shape functions and their derivatives
                el.CalcShape(ip, N);
                el.CalcDShape(ip, dN);

                // compute inverse transpose of J^T J
                DenseMatrix J = T.Jacobian();
                DenseMatrix JtJ_invT(J.NumCols());
                MultAtB(J, J, JtJ_invT);
                JtJ_invT.Invert(); JtJ_invT.Transpose();

                // compute derivative of Jacobian w.r.t. nodal coordinate
                Mult(dX_dXk, dN, dJ_dXk);

                // compute derivative of 0.5 J^T J
                DenseMatrix dJtJ_dXk(dim, dim);
                MultAtB(dJ_dXk, J, dJtJ_dXk);
                dJtJ_dXk.Symmetrize();

                // compute derivative of integration weight
                double dw_dXk = ip.weight * T.Weight() * MatrixInnerProduct(JtJ_invT, dJtJ_dXk);

                // compute kronecker
                DenseMatrix matN(N.GetData(), dof, 1);
                KroneckerProduct(I, matN, Kr_IN);

                // operator on traction vector
                f_->Eval(f, T, ip);
                Kr_IN.Mult(f, Kr_IN_f);

                // add integration point's derivative contribution
                dp_dXk.Add(dw_dXk, Kr_IN_f);
            }
            elvect(n+m*dof) += InnerProduct(ue_adjoint, dp_dXk);
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

void PhysicsSolverBase::UpdateMesh(Vector const &U)
{
  Vector Xi = X0_;
  Xi += U;
  coord_fes_->GetParMesh()->SetNodes(Xi);

  coord_fes_->GetParMesh()->DeleteGeometricFactors();
}

double QuantityOfInterest::EvalQoI()
{
  this->UpdateMesh(designVar);

  ParGridFunction oneGridFunction = ParGridFunction(temp_fes_);
  oneGridFunction = 1.0;
  ConstantCoefficient one(1.0);
  BilinearFormIntegrator *integ = nullptr;

  const int dim = pmesh->Dimension();


  switch (qoiType_) {
    case 7:
    {
      ParLinearForm loadForm(coord_fes_);
      if (tractionLoad_)
      {
        loadForm.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*tractionLoad_, 12, 12), bdr);
      }
      loadForm.Assemble();

      return loadForm(solgf_);
      break;
    }
    default:
      std::cout << "Unknown Error Coeff: " << qoiType_ << std::endl;
  }

  ParGridFunction ErrorGF = ParGridFunction(temp_fes_);

  ParLinearForm scalarErrorForm(temp_fes_);
  LFErrorIntegrator *lfi = new LFErrorIntegrator;
  lfi->SetQoI(ErrorCoefficient_);
  // lfi->SetIntRule(&IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  lfi->SetIntRule(&IntRulesGLL.Get(temp_fes_->GetFE(0)->GetGeomType(), 12));

  lfi->SetGLLVec(gllvec_);
  lfi->SetNqptsPerEl(nqptsperel);
  scalarErrorForm.AddDomainIntegrator(lfi);
  scalarErrorForm.Assemble();

  return scalarErrorForm(oneGridFunction);
}

void QuantityOfInterest::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ConstantCoefficient one(1.0);
  BilinearFormIntegrator *integ = nullptr;
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);

  if(qoiType_ == QoIType::STRUC_COMPLIANCE)
  {
    {
      ParLinearForm u_gradForm(coord_fes_);
      u_gradForm.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*tractionLoad_, 12, 12), bdr);
      u_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add(1.0, u_gradForm);
    }
    {
      ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddBoundaryIntegrator(new ElasticityTractionShapeSensitivityIntegrator(*tractionLoad_, solgf_, 12, 12), bdr);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);
    }
  }
}


void VectorHelmholtz::FSolve( )
{
  Vector Xi = X0_;
  coord_fes_->GetParMesh()->SetNodes(Xi);
  coord_fes_->GetParMesh()->DeleteGeometricFactors();

  Array<int> ess_tdof_list(ess_tdof_list_);

  ParBilinearForm a(temp_fes_);
  ParLinearForm b(temp_fes_);
  a.AddDomainIntegrator(new VectorMassIntegrator);
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(pradius_ ? *pradius_ : *radius_));

  b.AddDomainIntegrator(new VectorDomainLFIntegrator(*QCoef_));

  a.Assemble();
  b.Assemble();

  // solve for temperature
  ParGridFunction &u = solgf;

  u = 0.0;
  HypreParMatrix A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

  HypreBoomerAMG amg(A);
  amg.SetPrintLevel(0);

  CGSolver cg(coord_fes_->GetParMesh()->GetComm());
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(5000);
  cg.SetPreconditioner(amg);
  cg.SetOperator(A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, u);
}

void VectorHelmholtz::ASolve( Vector & rhs, bool isGradX )
{
    Vector Xi = X0_;
    coord_fes_->GetParMesh()->SetNodes(Xi);
    coord_fes_->GetParMesh()->DeleteGeometricFactors();

    Array<int> ess_tdof_list(ess_tdof_list_);

      // ParGridFunction bdrBoundIndicator(coord_fes_); bdrBoundIndicator = 0.0;
      // for(int i = 0; i<ess_tdof_list.Size(); i++)
      // {
      //    bdrBoundIndicator[ess_tdof_list[i]] =1.0;
      // }
      // ParaViewDataCollection paraview_dc_morph("isoel_bdr_morph", pmesh);
      // paraview_dc_morph.SetPrefixPath("ParaView");
      // paraview_dc_morph.SetLevelsOfDetail(1);
      // paraview_dc_morph.SetDataFormat(VTKFormat::BINARY);
      // paraview_dc_morph.SetHighOrderOutput(true);
      // paraview_dc_morph.SetCycle(0);
      // paraview_dc_morph.SetTime(0.0);
      // paraview_dc_morph.RegisterField("bdr", &bdrBoundIndicator);
      // paraview_dc_morph.Save();

    //ess_tdof_list.Print();

    ParBilinearForm a(temp_fes_);
    a.AddDomainIntegrator(new VectorMassIntegrator);
    a.AddDomainIntegrator(new VectorDiffusionIntegrator(pradius_ ? *pradius_ : *radius_));
    a.Assemble();

    // solve adjoint problem
    ParGridFunction adj_sol(temp_fes_);
    adj_sol = 0.0;

    HypreParMatrix A;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    HypreParMatrix* tTransOp = reinterpret_cast<HypreParMatrix*>(&A)->Transpose();

    HypreBoomerAMG amg(*tTransOp);
    amg.SetPrintLevel(0);

    CGSolver cg(coord_fes_->GetParMesh()->GetComm());
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(5000);
    cg.SetPreconditioner(amg);
    cg.SetOperator(*tTransOp);
    cg.Mult(B, X);

    delete tTransOp;

    a.RecoverFEMSolution(X, rhs, adj_sol);

    //adj_sol.Print();

    VectorGridFunctionCoefficient QF(&adj_sol);
    IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);

    if(isGradX)
    {
      ParLinearForm RHS_sensitivity(coord_fes_);
      VectorDomainLFIntegrator *lfi = new VectorDomainLFIntegrator(QF);

      lfi->SetIntRule(&IntRulesGLL.Get(coord_fes_->GetFE(0)->GetGeomType(), 24));
      RHS_sensitivity.AddDomainIntegrator(lfi);
      RHS_sensitivity.Assemble();

      *dQdx_ = 0.0;
      dQdx_->Add(1.0, RHS_sensitivity);   // - because
    }
    if( !isGradX )
    {
      {
        ParLinearForm RHS_sensitivity(temp_fes_scalar_);
        DomainLFGradIntegrator *lfi = new DomainLFGradIntegrator(QF);

        lfi->SetIntRule(&IntRulesGLL.Get(temp_fes_->GetFE(0)->GetGeomType(), 24));
        RHS_sensitivity.AddDomainIntegrator(lfi);
        RHS_sensitivity.Assemble();

        *dQdu_ = 0.0;
        dQdu_->Add(1.0, RHS_sensitivity);   // - because
      }

      //-------------------------------------------------------------------

      {
        ParLinearForm RHS_sensitivity(coord_fes_);
        LinearFormIntegrator *lfi = new GradProjectionShapeSensitivityIntegrator(solgf, adj_sol, *QCoef_);

        lfi->SetIntRule(&IntRulesGLL.Get(coord_fes_->GetFE(0)->GetGeomType(), 24));
        RHS_sensitivity.AddDomainIntegrator(lfi);
        RHS_sensitivity.Assemble();

        *dQdxshape_ = 0.0;
        dQdxshape_->Add(1.0, RHS_sensitivity);   // - because
      }
    }
}

void Elasticity_Solver::FSolve()
{
  this->UpdateMesh(designVar);

  Array<int> ess_tdof_list(ess_tdof_list_);

  ParBilinearForm a(physics_fes_);
  ParLinearForm b(physics_fes_);
  a.AddDomainIntegrator(new ElasticityIntegrator(*firstLameCoef, *secondLameCoef));
  b.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*QCoef_), bdr);

  a.Assemble();
  b.Assemble();

  // solve for temperature
  ParGridFunction &u = solgf;

  u = 0.0;
  HypreParMatrix A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

  HypreBoomerAMG amg(A);
  amg.SetPrintLevel(0);

  CGSolver cg(physics_fes_->GetParMesh()->GetComm());
  cg.SetRelTol(1e-10);
  cg.SetMaxIter(500);
  cg.SetPreconditioner(amg);
  cg.SetPrintLevel(0);
  cg.SetOperator(A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, u);
}

void Elasticity_Solver::ASolve( Vector & rhs )
{
    // the nodal coordinates will default to the initial mesh
    this->UpdateMesh(designVar);

    Array<int> ess_tdof_list(ess_tdof_list_);

    ParBilinearForm a(physics_fes_);
    a.AddDomainIntegrator(new ElasticityIntegrator(*firstLameCoef, *secondLameCoef));
    a.Assemble();

    // solve adjoint problem
    ParGridFunction adj_sol(physics_fes_);
    adj_sol = 0.0;

    HypreParMatrix A;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    HypreBoomerAMG amg(A);
    amg.SetPrintLevel(0);

    CGSolver cg(physics_fes_->GetParMesh()->GetComm());
    cg.SetRelTol(1e-10);
    cg.SetMaxIter(500);
    cg.SetPreconditioner(amg);
    cg.SetOperator(A);
    cg.Mult(B, X);

    a.RecoverFEMSolution(X, rhs, adj_sol);

    //adj_sol.Print();

    // make a Parlinear form to compute sensivity w.r.t. nodal coordinates
    // here we can use sensitivity w.r.t coordinate since d/dU = d/dX * dX/dU = d/dX * 1
    ParLinearForm LHS_sensitivity(coord_fes_);
    // LinearFormIntegrator *lfi = new ElasticityStiffnessShapeSensitivityIntegrator(
    //                                         *firstLameCoef, *secondLameCoef, solgf, adj_sol);
    // LHS_sensitivity.AddDomainIntegrator(lfi);
    LHS_sensitivity.AddDomainIntegrator(new ElasticityStiffnessShapeSensitivityIntegrator(
                                            *firstLameCoef, *secondLameCoef, solgf, adj_sol));
    LHS_sensitivity.Assemble();

    ParLinearForm RHS_sensitivity(coord_fes_);
    RHS_sensitivity.AddBoundaryIntegrator(new ElasticityTractionShapeSensitivityIntegrator(*QCoef_, adj_sol, 12,12), bdr);
    RHS_sensitivity.Assemble();

    *dQdx_ = 0.0;
    dQdx_->Add(-1.0, LHS_sensitivity);
    dQdx_->Add( 1.0, RHS_sensitivity);
}

DiffusionSolver::DiffusionSolver(Mesh * mesh_, int order_,
                                 Coefficient * diffcf_, Coefficient * rhscf_)
   : mesh(mesh_), order(order_), diffcf(diffcf_), rhscf(rhscf_)
{

#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif

   SetupFEM();
}

void DiffusionSolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver::Solve()
{
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   *u=0.0;
   if (b)
   {
      delete b;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif
   }
   if (rhscf)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*rhscf));
   }
   if (neumann_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*neumann_cf),neumann_bdr);
   }
   else if (gradient_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*gradient_cf),
                               neumann_bdr);
   }

   b->Assemble();

   BilinearForm * a = nullptr;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif
   a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
   if (masscf)
   {
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
   }
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B);

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
   }
   else
   {
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
   }
#else
   M = new GSSmoother((SparseMatrix&)(*A));
   cg = new CGSolver;
#endif
   cg->SetRelTol(1e-12);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *u);
   delete a;
}

mfem::GridFunction * DiffusionSolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI
mfem::ParGridFunction * DiffusionSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

DiffusionSolver::~DiffusionSolver()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
#ifdef MFEM_USE_MPI
   delete pfes; pfes=nullptr;
#endif
   delete fec; fec = nullptr;
   delete b;
}

}
