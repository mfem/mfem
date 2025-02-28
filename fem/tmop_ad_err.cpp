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

void KroneckerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, mfem::DenseMatrix &C)
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

void IsotropicStiffnessMatrix(int dim, double mu, double lambda, mfem::DenseMatrix &C)
{
    // set C = 2*mu*P_sym
    FourthOrderSymmetrizer(dim, C);
    C *= 2*mu;

    // compute lambda*dyadic(I, I)
    mfem::DenseMatrix I, IxI;
    mfem::Vector vecI;
    IdentityMatrix(dim, I);
    Vectorize(I, vecI);
    VectorOuterProduct(vecI, vecI, IxI);
    IxI *= lambda;

    // set C = 2*mu*P_sym + lamba*dyadic(I, I)
    C += IxI;
}

void IsotropicStiffnessMatrix3D(double E, double v, mfem::DenseMatrix &C)
{
    double     mu = E   / (2*(1+v));
    double lambda = E*v / ((1+v)*(1-2*v));
    IsotropicStiffnessMatrix(3, mu, lambda, C);
}

void VectorOuterProduct(const mfem::Vector &a, const mfem::Vector &b, mfem::DenseMatrix &C)
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

void FourthOrderSymmetrizer(int dim, mfem::DenseMatrix &S)
{
    mfem::DenseMatrix I, T;
    FourthOrderIdentity(dim, I);
    FourthOrderTranspose(dim, T);
    S  = I;
    S += T;
    S *= 0.5;
}

void FourthOrderIdentity(int dim, mfem::DenseMatrix &I4)
{
    mfem::DenseMatrix I2;
    IdentityMatrix(dim, I2);
    MatrixConjugationProduct(I2, I2, I4);
}

void FourthOrderTranspose(int dim, mfem::DenseMatrix &T)
{
    T.SetSize(dim*dim, dim*dim);
    T = 0.0;
    mfem::Vector Eij, Eji;
    mfem::DenseMatrix temp;
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

void UnitStrain(int dim, int i, int j, mfem::DenseMatrix &E)
{
    E.SetSize(dim, dim);
    E       = 0.0;
    E(i, j) = 1.0;
}

// E_ij = outer(e_i, e_j)
void UnitStrain(int dim, int i, int j, mfem::Vector &E)
{
    E.SetSize(dim*dim);
    E          = 0.0;
    E(j*dim+i) = 1.0;
}

void MatrixConjugationProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B, mfem::DenseMatrix &C)
{
    KroneckerProduct(B, A, C);
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

  int integrationOrder = 2 * el.GetOrder() + 3;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = IntRule;

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

void LFAverageErrorDerivativeIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
    mfem::Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  int integrationOrder = 2 * el.GetOrder() + 3;
  if (IntegrationOrder_ != INT_MAX) {
    integrationOrder = IntegrationOrder_;
  }

  // set integration rule
  const mfem::IntegrationRule *ir = IntRule;

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

  // /* should be this ->
  shapeSum = 0.0;
  double el_vol = 0.0;
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
      const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
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
  // const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

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

ThermalConductivityShapeSensitivityIntegrator_new::ThermalConductivityShapeSensitivityIntegrator_new(
  Coefficient &conductivity, const ParGridFunction &t_primal, const ParGridFunction &t_adjoint)
  : k_(&conductivity), t_primal_(&t_primal), t_adjoint_(&t_adjoint)
{}

void ThermalConductivityShapeSensitivityIntegrator_new::AssembleRHSElementVect(const FiniteElement &el,
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
  // const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * T.OrderGrad(&el));
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    double w = ip.weight * T.Weight();
    el.CalcDShape(ip, dN);
    Mult(dN, T.InverseJacobian(), B);

    Vector grad_primal(dim);
    Vector grad_adjoint(dim);

    t_primal_->GetGradient(T,grad_primal);
    t_adjoint_->GetGradient(T,grad_adjoint);

    double k = k_->Eval(T, ip);

    mfem::DenseMatrix GradAdjxGradPrimOuterProd;
    VectorOuterProduct(grad_adjoint, grad_primal, GradAdjxGradPrimOuterProd);
    Mult(B, GradAdjxGradPrimOuterProd, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( -2.0* w * k , IxBTvec);              // 2 times because of symmetric // TODO

    real_t val_3 = grad_primal * grad_adjoint;
    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * k * val_3, IxBTvec);
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
  Coefficient &t_primal, const ParGridFunction &t_adjoint, Coefficient &t_penalty, mfem::VectorCoefficient *SolGrad, int oa, int ob)
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
  //const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), oa_ * el.GetOrder() + ob_);
  const IntegrationRule *ir = IntRule;

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
        dp_dXk.Add( dw_dXk * Q_->Eval(T, ip), N);
      }
      elvect(n + m * dof) += InnerProduct(te_adjoint, dp_dXk);
    }
  }
}

ThermalHeatSourceShapeSensitivityIntegrator_new::ThermalHeatSourceShapeSensitivityIntegrator_new(Coefficient &heatSource,
    const ParGridFunction &t_adjoint,
    int oa, int ob)
  : Q_(&heatSource), t_adjoint_(&t_adjoint), oa_(oa), ob_(ob)
{}

void ThermalHeatSourceShapeSensitivityIntegrator_new::AssembleRHSElementVect(const FiniteElement &el,
    ElementTransformation &T, Vector &elvect)
{
  // grab sizes
  int dof = el.GetDof();
  int dim = el.GetDim();

  // initialize storage
  Vector N(dof);
  DenseMatrix dN(dof, dim);
  Vector te_adjoint(dof);
  DenseMatrix B(dof, dim);
  DenseMatrix IxB(dof, dim);
  Vector IxBTvec(dof * dim);

  // output vector
  elvect.SetSize(dim * dof);
  elvect = 0.0;

  DenseMatrix I;
  IdentityMatrix(dim, I);

  // set integration rule
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), oa_ * el.GetOrder() + ob_);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // set current integration point
    const IntegrationPoint &ip = ir->IntPoint(i);
    T.SetIntPoint(&ip);

    // evaluate gaussian integration weight
    double w = ip.weight * T.Weight();

    // evaluate shape function derivative
    el.CalcShape(ip, N);
    el.CalcDShape(ip, dN);

    // get inverse jacobian
    DenseMatrix Jinv = T.InverseJacobian();
    Mult(dN, Jinv, B);

    double adj_val = t_adjoint_->GetValue( T, ip);

    Mult(B, I, IxB);
    Vectorize(IxB, IxBTvec);
    elvect.Add( w * adj_val * Q_->Eval(T, ip), IxBTvec);

    // term2
    Vector loadgrad(dim);
    LoadGrad_->Eval(loadgrad, T, ip);
    for (int d = 0; d < dim; d++)
    {
      Vector elvect_temp(elvect.GetData() + d*dof, dof);
      elvect_temp.Add( w * adj_val * loadgrad(d), N); //k10-trial
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
    mfem::Coefficient &lambda, mfem::Coefficient &mu,
    const mfem::ParGridFunction &u_primal, const mfem::ParGridFunction &u_adjoint)
    : lambda_(&lambda), mu_(&mu), u_primal_(&u_primal), u_adjoint_(&u_adjoint)
{
    MFEM_ASSERT(u_primal.VectorDim() == u_adjoint.VectorDim(), "Primal and adjoint solutions are not compatible");
}

void ElasticityStiffnessShapeSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
        mfem::ElementTransformation &T,
        mfem::Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int dim = el.GetDim();

    // intialize intermediate matrices
    mfem::DenseMatrix                  dN(dof, dim);
    mfem::DenseMatrix                   B(dof, dim);
    mfem::DenseMatrix               Kr_IB(dim*dof, dim*dim);
    mfem::DenseMatrix                matC(dim*dim);
    mfem::DenseMatrix           KrIB_matC(dim*dof, dim*dim);

    mfem::DenseMatrix              dX_dXk(dim, dof);
    mfem::DenseMatrix              dJ_dXk(dim, dim);
    mfem::DenseMatrix           dJinv_dXk(dim, dim);
    mfem::DenseMatrix              dB_dXk(dof, dim);
    mfem::DenseMatrix              Kr_IdB(dim*dof, dim*dim);
    mfem::DenseMatrix          KrIdB_matC(dim*dof, dim*dim);
    mfem::DenseMatrix     KrIdB_matC_KrIB(dim*dof, dim*dof);
    mfem::DenseMatrix     KrIB_matC_KrIdB(dim*dof, dim*dof);
    mfem::DenseMatrix      KrIB_matC_KrIB(dim*dof, dim*dof);

    mfem::DenseMatrix              dK_dXk(dim*dof, dim*dof);

    mfem::Vector                ue_primal(dim*dof);
    mfem::Vector               ue_adjoint(dim*dof);

    mfem::Array<int> vdofs;
    u_primal_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
    u_primal_->GetSubVector(vdofs, ue_primal);

    u_adjoint_->ParFESpace()->GetElementVDofs(T.ElementNo, vdofs);
    u_adjoint_->GetSubVector(vdofs, ue_adjoint);

    // identity tensort
    mfem::DenseMatrix I;
    IdentityMatrix(dim, I);

    // output vector
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // set integration rule
    // const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), 2*T.OrderGrad(&el));
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &IntRulesGLL.Get(el.GetGeomType(), 8);

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
                const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
                T.SetIntPoint(&ip);                       // set current integration point
                double w = ip.weight * T.Weight();        // evaluate gaussian integration weight
                el.CalcDShape(ip, dN);                    // evaluate shape function derivative
                mfem::Mult(dN, T.InverseJacobian(), B);   // map to iso-parametric element
                KroneckerProduct(I, B, Kr_IB);  // compute Kron(B.T, I)
                IsotropicStiffnessMatrix(dim, mu_->Eval(T, ip), lambda_->Eval(T, ip), matC);

                // compute derivative of Jacobian w.r.t. nodal coordinate
                mfem::Mult(dX_dXk, dN, dJ_dXk);

                // compute derivative of J^(-1)
                mfem::DenseMatrix JinvT = T.InverseJacobian();
                JinvT.Transpose();
                ConjugationProduct(T.InverseJacobian(), JinvT, dJ_dXk, dJinv_dXk);
                dJinv_dXk *= -1.0;

                // compute derivative of B w.r.t. nodal coordinate
                mfem::Mult(dN, dJinv_dXk, dB_dXk);
                KroneckerProduct(I, dB_dXk, Kr_IdB);

                // compute derivative of stiffness matrix w.r.t. X_k
                mfem::Mult(Kr_IdB, matC, KrIdB_matC);
                mfem::Mult(Kr_IB, matC, KrIB_matC);
                mfem::MultABt(KrIdB_matC, Kr_IB, KrIdB_matC_KrIB);
                mfem::MultABt(KrIB_matC, Kr_IdB, KrIB_matC_KrIdB);
                mfem::MultABt(KrIB_matC, Kr_IB, KrIB_matC_KrIB);

                // compute derivative of integration weight w.r.t. X_k
                double dw_dXk = w * MatrixInnerProduct(JinvT, dJ_dXk);

                // put together all terms of product rule
                dK_dXk.Add(w, KrIdB_matC_KrIB);
                dK_dXk.Add(w, KrIB_matC_KrIdB);
                dK_dXk.Add(dw_dXk, KrIB_matC_KrIB);
            }
            elvect(n+m*dof) += dK_dXk.InnerProduct(ue_primal, ue_adjoint);
        }
    }
}

ElasticityTractionIntegrator::ElasticityTractionIntegrator(mfem::VectorCoefficient &f, int oa, int ob)
    : f_(&f), oa_(oa), ob_(ob)
{}

void ElasticityTractionIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T,
        mfem::Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int vdim = f_->GetVDim();

    // initialize storage
    mfem::Vector                        N(dof);
    mfem::DenseMatrix               Kr_IN(vdim*dof, vdim);
    mfem::Vector                        f(vdim);
    mfem::Vector                  Kr_IN_f(vdim*dof);

    // identity tensor
    mfem::DenseMatrix I;
    IdentityMatrix(vdim, I);

    // output vector
    elvect.SetSize(vdim*dof);
    elvect = 0.0;

    // set integration rule
    const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), oa_*el.GetOrder()+ob_);

    // loop over integration points
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        // set integration point
        const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
        T.SetIntPoint(&ip);

        // compute weight
        double w = ip.weight * T.Weight();

        // evaluate shape functions
        el.CalcShape(ip, N);

        // compute kronecker
        mfem::DenseMatrix matN(N.GetData(), dof, 1);
        KroneckerProduct(I, matN, Kr_IN);

        // operator on traction vector
        f_->Eval(f, T, ip);
        Kr_IN.Mult(f, Kr_IN_f);

        // add integration point's contribution
        elvect.Add(w, Kr_IN_f);
    }
}

ElasticityTractionShapeSensitivityIntegrator::ElasticityTractionShapeSensitivityIntegrator(
    mfem::VectorCoefficient &f, const mfem::ParGridFunction &u_adjoint, int oa, int ob)
    : f_(&f), u_adjoint_(&u_adjoint), oa_(oa), ob_(ob)
{}

void ElasticityTractionShapeSensitivityIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el,
        mfem::ElementTransformation &T,
        mfem::Vector &elvect)
{
    // grab sizes
    int dof = el.GetDof();
    int dim = el.GetDim();
    int vdim = f_->GetVDim();

    // initialize storage
    mfem::DenseMatrix              dX_dXk(vdim, dof);
    mfem::DenseMatrix              dJ_dXk(vdim, dim);
    mfem::Vector                        N(dof);
    mfem::DenseMatrix                  dN(dof, dim);
    mfem::DenseMatrix               Kr_IN(vdim*dof, vdim);
    mfem::Vector                        f(vdim);
    mfem::Vector                  Kr_IN_f(vdim*dof);
    mfem::Vector                   dp_dXk(vdim*dof);

    mfem::Vector               ue_adjoint(vdim*dof);
    mfem::Array<int> vdofs;
    u_adjoint_->ParFESpace()->GetBdrElementVDofs(T.ElementNo, vdofs);
    u_adjoint_->GetSubVector(vdofs, ue_adjoint);

    // identity tensor
    mfem::DenseMatrix I;
    IdentityMatrix(vdim, I);

    // output vector
    elvect.SetSize(vdim*dof);
    elvect = 0.0;

    // set integration rule
    const mfem::IntegrationRule *ir = &mfem::IntRules.Get(el.GetGeomType(), oa_*el.GetOrder()+ob_);

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
                const ::mfem::IntegrationPoint &ip = ir->IntPoint(i);
                T.SetIntPoint(&ip);

                // evaluate shape functions and their derivatives
                el.CalcShape(ip, N);
                el.CalcDShape(ip, dN);

                // compute inverse transpose of J^T J
                mfem::DenseMatrix J = T.Jacobian();
                mfem::DenseMatrix JtJ_invT(J.NumCols());
                mfem::MultAtB(J, J, JtJ_invT);
                JtJ_invT.Invert(); JtJ_invT.Transpose();

                // compute derivative of Jacobian w.r.t. nodal coordinate
                mfem::Mult(dX_dXk, dN, dJ_dXk);

                // compute derivative of 0.5 J^T J
                mfem::DenseMatrix dJtJ_dXk(dim, dim);
                mfem::MultAtB(dJ_dXk, J, dJtJ_dXk);
                dJtJ_dXk.Symmetrize();

                // compute derivative of integration weight
                double dw_dXk = ip.weight * T.Weight() * MatrixInnerProduct(JtJ_invT, dJtJ_dXk);

                // compute kronecker
                mfem::DenseMatrix matN(N.GetData(), dof, 1);
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
  ParGridFunction flux(coord_fes_);

  const int dim = pmesh->Dimension();
  mfem::L2_FECollection fecGF_0(0, dim, mfem::BasisType::GaussLobatto);
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
  mfem::VectorCoefficient * gradField = nullptr;
  mfem::VectorCoefficient * gradGF = nullptr;
  ParGridFunction fiteredGradField(coord_fes_);

  Error_QoI *ErrorCoefficient_T1_;
  H1SemiError_QoI *ErrorCoefficient_T2_;
  switch (qoiType_) {
  case 0:
     if( trueSolution_ == nullptr ){ mfem_error("true solution not set.");}
     true_solgf_.ProjectCoefficient(*trueSolution_);
     true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
      ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, trueSolutionGrad_);
    // ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, &true_solgradgf_coeff_);
     break;
  case 1:
    if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution not set.");}
    //  ErrorCoefficient_ = std::make_shared<H1SemiError_QoI>(&solgf_, trueSolutionGrad_, trueSolutionHess_);
     true_solgf_.ProjectCoefficient(*trueSolution_);
     true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
     MFEM_VERIFY(trueSolutionHess_ != nullptr, "trueSolutionHess_ is null");
     MFEM_VERIFY(trueSolutionHessV_ != nullptr, "trueSolutionHessV_ is null");
     true_solhessgf_.ProjectCoefficient(*trueSolutionHessV_);
     ErrorCoefficient_ = std::make_shared<H1SemiError_QoI>(&solgf_, trueSolutionGrad_, &true_solhessgf_coeff_);
    break;
  case 2:
    integ = new DiffusionIntegrator(one);
    solgf_.ComputeFlux(*integ, flux, false);
    trueSolutionGrad_ = new VectorGridFunctionCoefficient( &flux);
    ErrorCoefficient_ = std::make_shared<ZZError_QoI>(&solgf_, trueSolutionGrad_);
    break;
  case 3:
    ErrorCoefficient_ = std::make_shared<AvgError_QoI>(&solgf_, &L2Field, dim);
    break;
  case 4:
      if( trueSolution_ == nullptr ){ mfem_error("force coeff.");}
      ErrorCoefficient_ = std::make_shared<Energy_QoI>(&solgf_, trueSolution_, dim);
      break;
  case 5:
      if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution grad not set.");}
      {
        std::vector<std::pair<int, int>> essentialBC(0);
        double mesh_poly_deg = pmesh->GetNodalFESpace()->GetOrder(0);
        VectorHelmholtz filterSolver(pmesh, essentialBC, 0.0, mesh_poly_deg);
        gradGF= new mfem::GradientGridFunctionCoefficient(&solgf_);
        filterSolver.setLoadCoeff(gradGF);
        filterSolver.FSolve();
        fiteredGradField = filterSolver.GetSolution();

        gradField= new mfem::VectorGridFunctionCoefficient(&fiteredGradField);

        ErrorCoefficient_ = std::make_shared<ZZH1Error_QoI>(&solgf_, gradField);
      }
      break;
  case 6:
      // setup l2 first
      // true_solgf_.ProjectCoefficient(*trueSolution_);
      // true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
      //  ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, trueSolutionGrad_);
      ErrorCoefficient_T1_ = new Error_QoI(&solgf_, trueSolution_, &true_solgradgf_coeff_);

       true_solgf_.ProjectCoefficient(*trueSolution_);
       true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
       MFEM_VERIFY(trueSolutionHess_ != nullptr, "trueSolutionHess_ is null");
       MFEM_VERIFY(trueSolutionHessV_ != nullptr, "trueSolutionHessV_ is null");
       true_solhessgf_.ProjectCoefficient(*trueSolutionHessV_);
       ErrorCoefficient_T2_ = new H1SemiError_QoI(&solgf_, trueSolutionGrad_, &true_solhessgf_coeff_);
      // ErrorCoefficient_T2_ = new H1SemiError_QoI(&solgf_, trueSolutionGrad_, trueSolutionHess_);

      ErrorCoefficient_ = std::make_shared<H1Error_QoI>(ErrorCoefficient_T1_, ErrorCoefficient_T2_);
      break;
    case 7:
      // do nothing for now
      break;
  default:
    std::cout << "Unknown Error Coeff: " << qoiType_ << std::endl;
  }

  if(qoiType_ == QoIType::STRUC_COMPLIANCE)
  {
    ::mfem::ParLinearForm loadForm(coord_fes_);
    loadForm.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*tractionLoad_, 12, 12));
    loadForm.Assemble();

    return loadForm(solgf_);
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

  delete gradField;

  return scalarErrorForm(oneGridFunction);
}

void QuantityOfInterest::EvalQoIGrad()
{
  this->UpdateMesh(designVar);

  ConstantCoefficient one(1.0);
  BilinearFormIntegrator *integ = nullptr;
  ParGridFunction flux(coord_fes_);
  IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);

  const int dim = pmesh->Dimension();
  mfem::L2_FECollection fecGF_0(0, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fesGF_0(pmesh, &fecGF_0, 1);
  mfem::GridFunctionCoefficient tL2Coeff(&solgf_);
  mfem::ConstantCoefficient tConstCoeff(1.0);

  mfem::ParLinearForm b(&fesGF_0);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(tL2Coeff, 12, 12));
  b.Assemble();
  mfem::ParLinearForm b_Vol(&fesGF_0);
  b_Vol.AddDomainIntegrator(new mfem::DomainLFIntegrator(tConstCoeff, 12, 12));
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

  mfem::VectorCoefficient * gradField = nullptr;
  mfem::VectorCoefficient * gradGF = nullptr;
  ParGridFunction fiteredGradField(coord_fes_);

  std::vector<std::pair<int, int>> essentialBC(0);
  double mesh_poly_deg = pmesh->GetNodalFESpace()->GetMaxElementOrder();
  VectorHelmholtz filterSolver(pmesh, essentialBC, 0.0, mesh_poly_deg);

  Error_QoI *ErrorCoefficient_T1_;
  H1SemiError_QoI *ErrorCoefficient_T2_;

  switch (qoiType_) {
    case 0:
      if( trueSolution_ == nullptr ){ mfem_error("true solution not set.");}
     true_solgf_.ProjectCoefficient(*trueSolution_);
     true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);

       ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, trueSolutionGrad_);
      //  ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, &true_solgradgf_coeff_);
      break;
    case 1:
      if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution not set.");}
     true_solgf_.ProjectCoefficient(*trueSolution_);
     true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
     MFEM_VERIFY(trueSolutionHess_ != nullptr, "trueSolutionHess_ is null");
     MFEM_VERIFY(trueSolutionHessV_ != nullptr, "trueSolutionHessV_ is null");
     true_solhessgf_.ProjectCoefficient(*trueSolutionHessV_);
      ErrorCoefficient_ = std::make_shared<H1SemiError_QoI>(&solgf_, trueSolutionGrad_, &true_solhessgf_coeff_);
      //  ErrorCoefficient_ = std::make_shared<H1SemiError_QoI>(&solgf_, trueSolutionGrad_, trueSolutionHess_);
      break;
    case 2:
      integ = new DiffusionIntegrator(one);
      solgf_.ComputeFlux(*integ, flux, false);
      trueSolutionGrad_ = new VectorGridFunctionCoefficient( &flux);
      ErrorCoefficient_ = std::make_shared<ZZError_QoI>(&solgf_, trueSolutionGrad_);
      break;
    case 3:
      ErrorCoefficient_ = std::make_shared<AvgError_QoI>(&solgf_, &L2Field, dim);
      break;
    case 4:
      if( trueSolution_ == nullptr ){ mfem_error("force coeff.");}
      ErrorCoefficient_ = std::make_shared<Energy_QoI>(&solgf_, trueSolution_, dim);
      break;
    case 5:
      if( trueSolutionGrad_ == nullptr ){ mfem_error("true solution grad not set.");}
      {
        gradGF= new mfem::GradientGridFunctionCoefficient(&solgf_);
        filterSolver.setLoadCoeff(gradGF);
        filterSolver.FSolve();
        fiteredGradField = filterSolver.GetSolution();

        gradField= new mfem::VectorGridFunctionCoefficient(&fiteredGradField);

        ErrorCoefficient_ = std::make_shared<ZZH1Error_QoI>(&solgf_, gradField);
      }
      break;
  case 6:
      // setup l2 first
      // true_solgf_.ProjectCoefficient(*trueSolution_);
      // true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
      //  ErrorCoefficient_ = std::make_shared<Error_QoI>(&solgf_, trueSolution_, trueSolutionGrad_);
      ErrorCoefficient_T1_ = new Error_QoI(&solgf_, trueSolution_, &true_solgradgf_coeff_);
       true_solgf_.ProjectCoefficient(*trueSolution_);
       true_solgradgf_.ProjectCoefficient(*trueSolutionGrad_);
       MFEM_VERIFY(trueSolutionHess_ != nullptr, "trueSolutionHess_ is null");
       MFEM_VERIFY(trueSolutionHessV_ != nullptr, "trueSolutionHessV_ is null");
       true_solhessgf_.ProjectCoefficient(*trueSolutionHessV_);
       ErrorCoefficient_T2_ = new H1SemiError_QoI(&solgf_, trueSolutionGrad_, &true_solhessgf_coeff_);
       // ErrorCoefficient_T2_ = new H1SemiError_QoI(&solgf_, trueSolutionGrad_, trueSolutionHess_);
      ErrorCoefficient_ = std::make_shared<H1Error_QoI>(ErrorCoefficient_T1_, ErrorCoefficient_T2_);
      break;
    case 7:
      // do nothing for now
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
      // IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
      // lfi->SetIntRule(&IntRulesGLL.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      // lfi->SetIntRule(&IntRules.Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      lfi->SetIntRule(&irules->Get(temp_fes_->GetFE(0)->GetGeomType(), 8));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add( 1.0, T_gradForm);
    }
      // evaluate grad wrt coord
    {
      LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator;
      lfi->SetQoI(ErrorCoefficient_);
      // IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
      // lfi->SetIntRule(&IntRulesGLL.Get(coord_fes_->GetFE(0)->GetGeomType(), 8));
      lfi->SetIntRule(&irules->Get(coord_fes_->GetFE(0)->GetGeomType(), quad_order));

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

      lfi->SetIntRule(&irules->Get(coord_fes_->GetFE(0)->GetGeomType(), quad_order));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add( 1.0, T_gradForm);
    }
      // evaluate grad wrt coord
    {
      LFAvgErrorNodeCoordinateSensitivityIntegrator *lfi = new LFAvgErrorNodeCoordinateSensitivityIntegrator(&solgf_, &L2VolField);
      lfi->SetQoI(ErrorCoefficient_);

      lfi->SetIntRule(&irules->Get(coord_fes_->GetFE(0)->GetGeomType(), quad_order));

      ::mfem::ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddDomainIntegrator(lfi);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);
    }
  }
  else if(qoiType_ == QoIType::GZZ_ERROR)
  {
    // evaluate grad wrt temp
    {
      ParLinearForm T_gradForm(temp_fes_);
      LFErrorDerivativeIntegrator *lfi = new LFErrorDerivativeIntegrator;
      lfi->SetQoI(ErrorCoefficient_);

      lfi->SetIntRule(&irules->Get(temp_fes_->GetFE(0)->GetGeomType(), quad_order*4));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add(1.0, T_gradForm);

      //--------------------------------------------------------------------

      ParLinearForm SmoothTForm(coord_fes_);
      LFFilteredFieldErrorDerivativeIntegrator *lfi_s = new LFFilteredFieldErrorDerivativeIntegrator;
      lfi_s->SetQoI(ErrorCoefficient_);
      lfi_s->SetIntRule(&irules->Get(temp_fes_->GetFE(0)->GetGeomType(), quad_order*4));

      SmoothTForm.AddDomainIntegrator(lfi_s);
      SmoothTForm.Assemble();

      filterSolver.ASolve( SmoothTForm, false );

     // mfem::ParLinearForm * dQsdx = filterSolver.GetImplicitDqDxshape();

      mfem::ParLinearForm * dQsdu = filterSolver.GetImplicitDqDu();

      dQdu_->Add(-1.0, *dQsdu);

    }
    // evaluate grad wrt coord
    {
      LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator;
      lfi->SetQoI(ErrorCoefficient_);

      lfi->SetIntRule(&IntRulesGLL.Get(coord_fes_->GetFE(0)->GetGeomType(), 32));

      lfi->SetGLLVec(gllvec_);
      lfi->SetNqptsPerEl(nqptsperel);

      ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddDomainIntegrator(lfi);
      ud_gradForm.Assemble();
      *dQdx_ = 0.0;
      dQdx_->Add(1.0, ud_gradForm);

      // must be called after fitler.Asolve
      mfem::ParLinearForm * dQsdx = filterSolver.GetImplicitDqDxshape();
      dQdx_->Add(-1.0, *dQsdx);

    }

  }
  else if(qoiType_ == QoIType::STRUC_COMPLIANCE)
  {
    {
      ::mfem::ParLinearForm u_gradForm(coord_fes_);
      u_gradForm.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*tractionLoad_, 12, 12));
      u_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add(1.0, u_gradForm);
    }
    {
      ::mfem::ParLinearForm ud_gradForm(coord_fes_);
      ud_gradForm.AddBoundaryIntegrator(new ElasticityTractionShapeSensitivityIntegrator(*tractionLoad_, solgf_, 12, 12));
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

      lfi->SetIntRule(&irules->Get(temp_fes_->GetFE(0)->GetGeomType(), quad_order*4));
      T_gradForm.AddDomainIntegrator(lfi);
      T_gradForm.Assemble();
      *dQdu_ = 0.0;
      dQdu_->Add(1.0, T_gradForm);
    }

    // evaluate grad wrt coord
    {
      LFNodeCoordinateSensitivityIntegrator *lfi = new LFNodeCoordinateSensitivityIntegrator;
      lfi->SetQoI(ErrorCoefficient_);
      lfi->SetIntRule(&IntRulesGLL.Get(coord_fes_->GetFE(0)->GetGeomType(), 32));

      lfi->SetGLLVec(gllvec_);
      lfi->SetNqptsPerEl(nqptsperel);

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

    Array<int> ess_tdof_list;
    if(!weakBC_)
    {
      ess_tdof_list=ess_tdof_list_;
    }

  // assemble LHS matrix
  ConstantCoefficient kCoef(1.0);
  ConstantCoefficient wCoef(1e5);
  ProductCoefficient  truesolWCoef(wCoef,*trueSolCoeff);

  ParBilinearForm kForm(physics_fes_);
  ParLinearForm QForm(physics_fes_);
  kForm.AddDomainIntegrator(new DiffusionIntegrator(kCoef));

  QForm.AddDomainIntegrator(new DomainLFIntegrator(*QCoef_, 22,22));

  if(weakBC_)
  {
    kForm.AddBoundaryIntegrator(new BoundaryMassIntegrator(wCoef));

    QForm.AddBoundaryIntegrator(new BoundaryLFIntegrator(truesolWCoef, 12, 12));
  }

  kForm.Assemble();
  QForm.Assemble();

  solgf = 0.0;

  // solve for temperature
  ParGridFunction &T = solgf;
  if (trueSolCoeff)
  {
    //T.ProjectBdrCoefficient(*trueSolCoeff, ess_bdr_attr);
  }

  HypreParMatrix A;
  Vector X, B;
  kForm.FormLinearSystem(ess_tdof_list, T, QForm, A, X, B);

  HypreBoomerAMG amg(A);
  amg.SetPrintLevel(0);

  CGSolver cg(physics_fes_->GetParMesh()->GetComm());
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(5000);
  cg.SetPreconditioner(amg);
  cg.SetOperator(A);
  cg.SetPrintLevel(0);

  cg.Mult(B, X);

  kForm.RecoverFEMSolution(X, QForm, T);
}

void Diffusion_Solver::ASolve( Vector & rhs )
{
    // the nodal coordinates will default to the initial mesh
    this->UpdateMesh(designVar);

    ParGridFunction adj_sol(physics_fes_);
    ParGridFunction dQdu(physics_fes_);
    adj_sol = 0.0;
    ParGridFunction rhs_gf(physics_fes_);
    rhs_gf = 1.0;
    rhs_gf.ExchangeFaceNbrData();
    rhs_gf.SetTrueVector();
    rhs_gf.SetFromTrueVector();
    Vector &rhs_tvec = rhs_gf.GetTrueVector();

    bool visualization = false;
    ParGridFunction tempgf(coord_fes_);
    tempgf = 0.0;
    // VisItDataCollection vis_dc("Adjoint", physics_fes_->GetParMesh());
    // vis_dc.SetLevelsOfDetail(physics_fes_->GetMaxElementOrder()-1);
    ParaViewDataCollection vis_dc("Adjoint", physics_fes_->GetParMesh());
    vis_dc.SetLevelsOfDetail(physics_fes_->GetMaxElementOrder());
    vis_dc.SetHighOrderOutput(true);
    if (visualization)
    {
      vis_dc.SetCycle(pdc_cycle);
      vis_dc.SetTime((pdc_cycle++)*1.0);
      adj_sol = rhs;
      vis_dc.RegisterField("adjoint",&adj_sol);
      vis_dc.RegisterField("sensitivities",&tempgf);
      vis_dc.Save();
      adj_sol = 0.0;
    }

    Array<int> ess_tdof_list;
    if(!weakBC_)
    {
      ess_tdof_list=ess_tdof_list_;
    }

    //copy BC values from the grid function to the solution vector
    if(!weakBC_)
    {
      for(int ii=0;ii<ess_tdof_list.Size();ii++)
      {
        rhs_tvec[ess_tdof_list[ii]] = 0.0;
      }
      rhs_gf.SetFromTrueDofs(rhs_tvec);
      for (int ii = 0; ii < rhs.Size(); ii++)
      {
        rhs(ii) *= rhs_gf(ii); // zero out rhs for true dofs
      }
    }

    // assemble LHS matrix
    ConstantCoefficient kCoef(1.0);
    ConstantCoefficient wCoef(1e5);
    ProductCoefficient  truesolWCoef(wCoef,*trueSolCoeff);

    ParBilinearForm kForm(physics_fes_);
    kForm.AddDomainIntegrator(new DiffusionIntegrator(kCoef));
    if(weakBC_)
    {
      kForm.AddBoundaryIntegrator(new BoundaryMassIntegrator(wCoef));
    }
    kForm.Assemble();
    //kForm.Finalize();

    // solve adjoint problem
    HypreParMatrix A;
    Vector X, B;
    kForm.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    mfem::HypreParMatrix* tTransOp = reinterpret_cast<mfem::HypreParMatrix*>(&A)->Transpose();

    HypreBoomerAMG amg(*tTransOp);
    amg.SetPrintLevel(0);

    CGSolver cg(physics_fes_->GetParMesh()->GetComm());
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(5000);
    cg.SetPreconditioner(amg);
    cg.SetOperator(*tTransOp);
    // cg.SetPrintLevel(2);
    cg.Mult(B, X);
    kForm.RecoverFEMSolution(X, rhs, adj_sol);

    if (visualization)
    {
      vis_dc.SetCycle(pdc_cycle);
      vis_dc.SetTime((pdc_cycle++)*1.0);
      vis_dc.Save();
    }

    delete tTransOp;

    ParLinearForm LHS_sensitivity(coord_fes_);
    LHS_sensitivity.AddDomainIntegrator(new ThermalConductivityShapeSensitivityIntegrator_new(kCoef, solgf, adj_sol));
    if(weakBC_)
    {
      LHS_sensitivity.AddBoundaryIntegrator(new PenaltyMassShapeSensitivityIntegrator(wCoef, solgf, adj_sol));
    }
    LHS_sensitivity.Assemble();

    ParLinearForm RHS_sensitivity(coord_fes_);
    ThermalHeatSourceShapeSensitivityIntegrator_new *lfi = new ThermalHeatSourceShapeSensitivityIntegrator_new(*QCoef_, adj_sol);
    ParGridFunction trueSolutionGrad_gf(coord_fes_);
    trueSolutionGrad_gf.ProjectCoefficient(*trueSolutionGradCoef_);
    VectorGridFunctionCoefficient trueSolGradCoeff_Discrete(&trueSolutionGrad_gf);
    if(weakBC_)
    {
      MFEM_VERIFY(trueSolutionGradCoef_, "Set true solution gradient for boundary rhs sensitivity\n" );
      RHS_sensitivity.AddBoundaryIntegrator(new PenaltyShapeSensitivityIntegrator(*trueSolCoeff, adj_sol, wCoef, trueSolutionGradCoef_, 12, 12));
      // RHS_sensitivity.AddBoundaryIntegrator(new PenaltyShapeSensitivityIntegrator(*trueSolCoeff, adj_sol, wCoef, &trueSolGradCoeff_Discrete, 12, 12));
    }
    if (loadGradCoef_) {
      lfi->SetLoadGrad(loadGradCoef_);
      // trueloadgradgf_.ProjectCoefficient(*loadGradCoef_);
      // lfi->SetLoadGrad(&trueloadgradgf_coeff_);
    }
    IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
    lfi->SetIntRule(&IntRulesGLL.Get(physics_fes_->GetFE(0)->GetGeomType(), 24));
    RHS_sensitivity.AddDomainIntegrator(lfi);
    RHS_sensitivity.Assemble();

    if (visualization)
    {
      HypreParVector *tvec = LHS_sensitivity.ParallelAssemble();
      tempgf.SetFromTrueDofs(*tvec);
      vis_dc.SetCycle(pdc_cycle);
      vis_dc.SetTime((pdc_cycle++)*1.0);
      vis_dc.Save();

      HypreParVector *tvec2 = RHS_sensitivity.ParallelAssemble();
      tempgf.SetFromTrueDofs(*tvec2);
      vis_dc.SetCycle(pdc_cycle);
      vis_dc.SetTime((pdc_cycle++)*1.0);
      vis_dc.Save();

      MFEM_ABORT(" ");
    }

    *dQdx_ = 0.0;
    dQdx_->Add(-1.0, LHS_sensitivity);
    dQdx_->Add( 1.0, RHS_sensitivity);
}

void VectorHelmholtz::FSolve( )
{
  Vector Xi = X0_;
  coord_fes_->GetParMesh()->SetNodes(Xi);
  coord_fes_->GetParMesh()->DeleteGeometricFactors();

  Array<int> ess_tdof_list(ess_tdof_list_);

  ParBilinearForm a(coord_fes_);
  ParLinearForm b(coord_fes_);
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

    ParBilinearForm a(coord_fes_);
    a.AddDomainIntegrator(new VectorMassIntegrator);
    a.AddDomainIntegrator(new VectorDiffusionIntegrator(pradius_ ? *pradius_ : *radius_));
    a.Assemble();

    // solve adjoint problem
    ParGridFunction adj_sol(coord_fes_);
    adj_sol = 0.0;

    HypreParMatrix A;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, adj_sol, rhs, A, X, B);

    mfem::HypreParMatrix* tTransOp = reinterpret_cast<mfem::HypreParMatrix*>(&A)->Transpose();

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

    ::mfem::VectorGridFunctionCoefficient QF(&adj_sol);
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
        ParLinearForm RHS_sensitivity(temp_fes_);
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

  // make coefficients of the linear elastic properties
  ::mfem::ConstantCoefficient firstLameCoef(1.0);
  ::mfem::ConstantCoefficient secondLameCoef(0.0);

  ParBilinearForm a(physics_fes_);
  ParLinearForm b(physics_fes_);
  a.AddDomainIntegrator(new ElasticityIntegrator(firstLameCoef, secondLameCoef));
  b.AddBoundaryIntegrator(new ElasticityTractionIntegrator(*QCoef_));

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
  cg.SetOperator(A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, u);
}

void Elasticity_Solver::ASolve( Vector & rhs )
{
    // the nodal coordinates will default to the initial mesh
    this->UpdateMesh(designVar);

    Array<int> ess_tdof_list(ess_tdof_list_);

    // make coefficients of the linear elastic properties
    ::mfem::ConstantCoefficient firstLameCoef(1.0);
    ::mfem::ConstantCoefficient secondLameCoef(0.0);

    ParBilinearForm a(physics_fes_);
    a.AddDomainIntegrator(new ElasticityIntegrator(firstLameCoef, secondLameCoef));
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

    // make a Parlinear form to compute sensivity w.r.t. nodal coordinates
    // here we can use sensitivity w.r.t coordinate since d/dU = d/dX * dX/dU = d/dX * 1
    ::mfem::ParLinearForm LHS_sensitivity(coord_fes_);
    LHS_sensitivity.AddDomainIntegrator(new ElasticityStiffnessShapeSensitivityIntegrator(
                                            firstLameCoef, secondLameCoef, solgf, adj_sol));
    LHS_sensitivity.Assemble();

    ::mfem::ParLinearForm RHS_sensitivity(coord_fes_);
    RHS_sensitivity.AddBoundaryIntegrator(new ElasticityTractionShapeSensitivityIntegrator(*QCoef_, adj_sol, 12,12));
    RHS_sensitivity.Assemble();

    *dQdx_ = 0.0;
    dQdx_->Add(-1.0, LHS_sensitivity);
    dQdx_->Add( 1.0, RHS_sensitivity);
}

}
