
using namespace std;
using namespace mfem;

void BasisEval(const Vector xi, Vector &N, DenseMatrix &dNdxi) // dNdxi is 2*4
{
   N[0] = 0.25*(1-xi[0])*(1-xi[1]);
   N[1] = 0.25*(1+xi[0])*(1-xi[1]);
   N[2] = 0.25*(1+xi[0])*(1+xi[1]);
   N[3] = 0.25*(1-xi[0])*(1+xi[1]);

   dNdxi(0,0) = 0.25*(-1+xi[1]);
   dNdxi(0,1) = 0.25*(1-xi[1]);
   dNdxi(0,2) = 0.25*(1+xi[1]);
   dNdxi(0,3) = 0.25*(-1-xi[1]);
   dNdxi(1,0) = 0.25*(-1+xi[0]);
   dNdxi(1,1) = 0.25*(-1-xi[0]);
   dNdxi(1,2) = 0.25*(1+xi[0]);
   dNdxi(1,3) = 0.25*(1-xi[0]);
}


void BasisEvalDerivs(const Vector xi, Vector& N, DenseMatrix& dNdxi,
                     DenseMatrix& dN2dxi)
{
   N[0] = 0.25*(1-xi[0])*(1-xi[1]);
   N[1] = 0.25*(1+xi[0])*(1-xi[1]);
   N[2] = 0.25*(1+xi[0])*(1+xi[1]);
   N[3] = 0.25*(1-xi[0])*(1+xi[1]);

   dNdxi.SetSize(2,4); dNdxi = 0.0;
   dN2dxi.SetSize(3,4);
   dN2dxi = 0.0;  // first row dxi2, second detadxi, third deta2

   dNdxi(0,0) = 0.25*(-1+xi[1]); dNdxi(0,1) = 0.25*(1-xi[1]);
   dNdxi(0,2) = 0.25*(1+xi[1]);  dNdxi(0,3) = 0.25*(-1-xi[1]);
   dNdxi(1,0) = 0.25*(-1+xi[0]); dNdxi(1,1) = 0.25*(-1-xi[0]);
   dNdxi(1,2) = 0.25*(1+xi[0]);  dNdxi(1,3) = 0.25*(1-xi[0]);

   dN2dxi(1,0) = 0.25; dN2dxi(1,1) = -0.25; dN2dxi(1,2) = 0.25;
   dN2dxi(1,3) = -0.25;
}

// returns the vector and matrix form of the shape functions and its derivative
void BasisVectorDerivs(const Vector xi, DenseMatrix& N, DenseMatrix& dNdxi,
                       DenseMatrix& ddNdxi)
{
   N.SetSize(3,12); N = 0.0;
   N(0,0) = 0.25*(1-xi[0])*(1-xi[1]); N(0,3) = 0.25*(1+xi[0])*(1-xi[1]);
   N(0,6) = 0.25*(1+xi[0])*(1+xi[1]); N(0,9) = 0.25*(1-xi[0])*(1+xi[1]);

   N(1,1) = 0.25*(1-xi[0])*(1-xi[1]); N(1,4) = 0.25*(1+xi[0])*(1-xi[1]);
   N(1,7) = 0.25*(1+xi[0])*(1+xi[1]); N(1,10) = 0.25*(1-xi[0])*(1+xi[1]);

   N(2,2) = 0.25*(1-xi[0])*(1-xi[1]); N(2,5) = 0.25*(1+xi[0])*(1-xi[1]);
   N(2,8) = 0.25*(1+xi[0])*(1+xi[1]); N(2,11) = 0.25*(1-xi[0])*(1+xi[1]);

   dNdxi.SetSize(3*2, 3*4);  dNdxi = 0.0;
   dNdxi(0,0) = 0.25*(-1+xi[1]);  dNdxi(0,3) = 0.25*(1-xi[1]);
   dNdxi(0,6) = 0.25*(1+xi[1]);  dNdxi(0,9) = 0.25*(-1-xi[1]);
   dNdxi(1,1) = 0.25*(-1+xi[1]);  dNdxi(1,4) = 0.25*(1-xi[1]);
   dNdxi(1,7) = 0.25*(1+xi[1]);  dNdxi(1,10) = 0.25*(-1-xi[1]);
   dNdxi(2,2) = 0.25*(-1+xi[1]);  dNdxi(2,5) = 0.25*(1-xi[1]);
   dNdxi(2,8) = 0.25*(1+xi[1]);  dNdxi(2,11) = 0.25*(-1-xi[1]);

   dNdxi(3,0) = 0.25*(-1+xi[0]);  dNdxi(3,3) = 0.25*(-1-xi[0]);
   dNdxi(3,6) = 0.25*(1+xi[0]);  dNdxi(3,9) = 0.25*(1-xi[0]);
   dNdxi(4,1) = 0.25*(-1+xi[0]);  dNdxi(4,4) = 0.25*(-1-xi[0]);
   dNdxi(4,7) = 0.25*(1+xi[0]);  dNdxi(4,10) = 0.25*(1-xi[0]);
   dNdxi(5,2) = 0.25*(-1+xi[0]);  dNdxi(5,5) = 0.25*(-1-xi[0]);
   dNdxi(5,8) = 0.25*(1+xi[0]);  dNdxi(5,11) = 0.25*(1-xi[0]);

   ddNdxi.SetSize(3*4, 3*4); ddNdxi = 0.0;
   ddNdxi(3,0) = 0.25; ddNdxi(3,3) = -0.25;
   ddNdxi(3,6) = 0.25; ddNdxi(3,9) = -0.25;
   ddNdxi(4,1) = 0.25; ddNdxi(4,4) = -0.25;
   ddNdxi(4,7) = 0.25; ddNdxi(4,10) = -0.25;
   ddNdxi(5,2) = 0.25; ddNdxi(5,5) = -0.25;
   ddNdxi(5,8) = 0.25; ddNdxi(5,11) = -0.25;

   ddNdxi(6,0) = 0.25; ddNdxi(6,3) = -0.25;
   ddNdxi(6,6) = 0.25; ddNdxi(6,9) = -0.25;
   ddNdxi(7,1) = 0.25; ddNdxi(7,4) = -0.25;
   ddNdxi(7,7) = 0.25; ddNdxi(7,10) = -0.25;
   ddNdxi(8,2) = 0.25; ddNdxi(8,5) = -0.25;
   ddNdxi(8,8) = 0.25; ddNdxi(8,11) = -0.25;
}


void cross(const Vector a, const Vector b, Vector& c)
{
   assert(a.Size()==3);
   c.SetSize(3);
   c[0] = a[1]*b[2] - a[2]*b[1];
   c[1] = -a[0]*b[2] + b[0]*a[2];
   c[2] = a[0]*b[1] - a[1]*b[0];

}
// a outer b
void outer(const Vector a, const Vector b, DenseMatrix& c)
{
   int m = a.Size();
   int n = b.Size();
   assert(c.Height()==m);
   assert(c.Width() ==n);
   for (int i=0; i<m; i++)
   {
      for (int j=0; j<n; j++)
      {
         c(i,j) = a[i]*b[j];
      }
   }
}
// dphidxi 2*4
// coords 4*3
void ComputeNormal(const DenseMatrix& dphidxi, const DenseMatrix& coords,
                   Vector& normal, double& nnorm)
{

   DenseMatrix dxdxi(2,3);
   Mult(dphidxi, coords, dxdxi);
   Vector dxdxi1(3);
   Vector dxdxi2(3);

   dxdxi.GetRow(0,dxdxi1);
   dxdxi.GetRow(1,dxdxi2);

   cross(dxdxi1, dxdxi2, normal);  // is there a cross product?  no
   // VectorCrossProductCoefficient::Eval has hard-coded cross product
   nnorm = normal.Norml2( );
   normal /=  nnorm;
}

void SlaveToMaster(const DenseMatrix& m_coords, const Vector& s_x, Vector& xi)
{
   bool converged = false;
   bool pt_on_elem = false;
   int dim = 3;
   xi.SetSize(dim-1);
   xi = 0.0;
   int max_iter = 15;
   double off_el_xi = 1e-2;
   double proj_newton_tol = 1e-13;
   double proj_max_gap = 0.5;
   Vector gap_v(dim);
   // warm start from linear solution

   for (int it=0; it<max_iter; it++)
   {
      //cout<<it<<endl;
      Vector m_N(4);
      m_N = 0.;
      DenseMatrix m_dN(2,4);
      m_dN = 0.;
      DenseMatrix m_dN2(3,4);
      m_dN2 = 0.;
      BasisEvalDerivs(xi, m_N, m_dN, m_dN2);

      Vector x_c(dim);
      m_coords.MultTranspose(m_N, x_c);

      gap_v = s_x;
      gap_v -= x_c;

      DenseMatrix m_dx(2,3);
      m_dx = 0.;
      Mult(m_dN, m_coords, m_dx);

      Vector r(dim-1);
      r = 0.0;
      m_dx.Mult(gap_v, r);

      if (r.Normlinf() < proj_newton_tol)
      {
         converged = true;
         break;
      }

      DenseMatrix drdxi(dim-1,dim-1);
      drdxi = 0.;
      MultABt(m_dx, m_dx, drdxi); // m_dx * m_dx.T
      drdxi *= -1.0;

      DenseMatrix m_dx2(3,3); m_dx2 = 0.0;
      Mult(m_dN2,m_coords, m_dx2);

      //m_d2x = m_dN(:,:,2) * m_elem_coords(1:4,:);  //m_dN(:,:,2) is 3*4
      for (int d=0; d<3; d++)
      {
         DenseMatrix Mtemp(2,2); Mtemp = 0.0;
         Mtemp(0,0) = m_dx2(0,d); Mtemp(0,1) = m_dx2(1,d);
         Mtemp(1,0) = m_dx2(1,d); Mtemp(1,1) = m_dx2(2,d);

         drdxi.Add(gap_v[d], Mtemp);
      }

      //cond_num = rcond(drdxi);  condition number?
      //drdxi.TestInversion();
      DenseMatrixInverse drdxi_inv(drdxi);
      Vector xi_tmp(dim-1);

      drdxi_inv.Mult(r,xi_tmp);
      xi -= xi_tmp;
   }
   if (!converged)
   {
      xi = 0.0;
   }
   off_el_xi += 1 ; // tolerance of offset of xi outside [-1,1]

   //cout<<gap_v.Norml2()<<" " <<xi.Normlinf()<<endl;
   //
   // Discuss with Frank... what is happening here
   if (gap_v.Norml2() < proj_max_gap && xi.Normlinf() <= off_el_xi)
   {
      pt_on_elem = true;
   }

   if (pt_on_elem)
   {
     //cout << "convergence of node to segment projection? " << converged << endl;
     //for(int i = 0; i < 2; i++)
     //{
     //  cout << "xi_" << i << " = " << xi(i) << endl;
     //}
   }
   MFEM_VERIFY(pt_on_elem == true, "xi went out of bounds");
   MFEM_VERIFY(converged == true, "projection didn't converge");
}



// m_coords is expected to be 4 * 3
void  ComputeGapJacobian(const Vector x_s, const Vector xi,
                         const DenseMatrix m_coords,
                         double& gap, Vector& normal, Vector& dgdxm, Vector& dgdxs)
{
   Vector m_N(4);
   DenseMatrix m_dN(2,4);
   DenseMatrix m_dN2(3,4);
   BasisEvalDerivs(xi, m_N, m_dN, m_dN2);

   Vector x_c(3);
   m_coords.MultTranspose(m_N, x_c);

   Vector gap_v(3); gap_v = 0.0;
   gap_v  = x_s;
   gap_v -= x_c;

   DenseMatrix m_dx(2,3);
   Mult(m_dN, m_coords, m_dx);

   double nnorm = 0;
   ComputeNormal(m_dN, m_coords, normal, nnorm);

   gap = gap_v * normal; // gap function value, dot product between vectors

   //dr_dx = zeros(2,4,3); % nsegment, nodes in quad, ndim

   DenseMatrix dr_dx_res1(4,3); dr_dx_res1 = 0.;
   DenseMatrix dr_dx_res2(4,3); dr_dx_res2 = 0.;

   Vector m_dxrow1(3);
   m_dx.GetRow(0, m_dxrow1);
   outer(m_N, m_dxrow1, dr_dx_res1);// 4*1 times 1*3
   dr_dx_res1 *= -1.0;

   Vector m_dxrow2(3);
   m_dx.GetRow(1, m_dxrow2);
   outer(m_N, m_dxrow2, dr_dx_res2);// 4*1 times 1*3
   dr_dx_res2 *= -1.0;

   Vector m_dNrow1(4); m_dN.GetRow(0, m_dNrow1);
   Vector m_dNrow2(4); m_dN.GetRow(1, m_dNrow2);

   DenseMatrix dr_dx_res1_tmp(4,3); dr_dx_res1_tmp = 0.;
   DenseMatrix dr_dx_res2_tmp(4,3); dr_dx_res2_tmp = 0.;
   outer(m_dNrow1, gap_v, dr_dx_res1_tmp);// 4*1 times 1*3
   outer(m_dNrow2, gap_v, dr_dx_res2_tmp);// 4*1 times 1*3

   dr_dx_res1 += dr_dx_res1_tmp; // outer product in vector?
   dr_dx_res2 += dr_dx_res2_tmp;


   DenseMatrix K_dxidx1(2,2); // 2*2
   K_dxidx1 = 0.;
   MultABt(m_dx, m_dx, K_dxidx1); // m_dx * m_dx.T

   Vector v_dxidx2(4);
   m_coords.Mult(gap_v, v_dxidx2); // m_coords * gap_v;  // 4*3 * 3 = 4

   DenseMatrix K_dxidx2(2,2); K_dxidx2 = 0.0;

   Vector m_dN2row1(4); m_dN2.GetRow(0, m_dN2row1);
   Vector m_dN2row2(4); m_dN2.GetRow(1, m_dN2row2);
   Vector m_dN2row3(4); m_dN2.GetRow(2, m_dN2row3);
   // how to get 2nd order? multidimensional matrix?
   K_dxidx2(0,0) = m_dN2row1 * v_dxidx2; // how would 4*1 * 1*4 be computed?
   K_dxidx2(0,1) = m_dN2row2 * v_dxidx2;
   K_dxidx2(1,0) = m_dN2row2 * v_dxidx2;
   K_dxidx2(1,1) = m_dN2row3 * v_dxidx2;

   DenseMatrix K_dxidx(2,2);
   K_dxidx -= K_dxidx1;
   K_dxidx += K_dxidx2;

   // resize the vectors and matrices
   Vector dxidx(24); dxidx = 0.0;
   Vector drdx_r(24); drdx_r = 0.0;

   for (int i=0; i<4; i++)
   {
      for (int j=0; j<3; j++)
      {
         drdx_r[4*j+i] = dr_dx_res1(i,j);
         drdx_r[4*j+i+12] = dr_dx_res2(i,j);

      }
   }
   //drdx_r(1:4*3,1) = reshape(dr_dx_res(:,:,1),4*3,1);
   //drdx_r(4*3+1:2*4*3,1) = reshape(dr_dx_res(:,:,2),4*3,1);
   DenseMatrix drdx_K(24,24); drdx_K = 0.;
   for (int i =0; i<12; i++)
   {
      drdx_K(i,i) = K_dxidx(0,0);
      drdx_K(i,12+i) = K_dxidx(0,1);
      drdx_K(12+i,i) = K_dxidx(1,0);
      drdx_K(12+i,12+i) = K_dxidx(1,1);
   }

   DenseMatrixInverse drdxK_inv(drdx_K);
   drdxK_inv.Mult(drdx_r,dxidx);
   // LinearSolve (drdx_K,drdx_r, dxidx) ; //???
   dxidx *= -1.0;



   Vector drdxs_r(6);
   drdxs_r[0] = m_dx(0,0); drdxs_r[1] = m_dx(0,1); drdxs_r[2] = m_dx(0,2);
   drdxs_r[3] = m_dx(1,0); drdxs_r[4] = m_dx(1,1); drdxs_r[5] = m_dx(1,2);

   DenseMatrix drdxs_K(6,6); drdxs_K = 0.;
   for (int i=0; i<3; i++)
   {
      drdxs_K(i,i) = K_dxidx(0,0);
      drdxs_K(i,3+i) = K_dxidx(0,1);
      drdxs_K(i+3,i) = K_dxidx(1,0);
      drdxs_K(i+3,i+3) = K_dxidx(1,1);
   }

   Vector dxidxs(6); dxidxs = 0.0;
   DenseMatrixInverse drdxsK_inv(drdxs_K);
   drdxsK_inv.Mult(drdxs_r,dxidxs);
   dxidxs *= -1.0;
   //dxidxs = -drdxs_K\drdxs_r;

   //dxidx = reshape(dxidx, 4,3,2); dxidxs = reshape(dxidxs, 1,3,2);

   dgdxm.SetSize(12); dgdxm = 0.;
   DenseMatrix dgdxm_tmp(4,3);
   outer(m_N, normal,dgdxm_tmp);
   for (int i=0; i<4; i++)
   {
      for (int j=0; j<3; j++)
      {
         dgdxm[3*i+j] = -dgdxm_tmp(i,j);
      }
   }
   //dxidx_M = -m_dN(1:2,:,1) * (m_coords(1:4,:)*normal'); % this turns out to be 0

   dgdxs.SetSize(3);
   dgdxs += normal;
   //dgdxs = dgdxs + dxidx_M(1) * dxidxs(:,:,1) + dxidx_M(2) * dxidxs(:,:,2);
};

void ComputeGapHessian(const Vector x_s, const Vector xi,
                       const DenseMatrix m_coords,
                       DenseMatrix& dg2dx)
{
   Vector m_N(4);
   DenseMatrix m_dN(2,4);
   DenseMatrix m_dN2(3,4);
   BasisEvalDerivs(xi, m_N, m_dN, m_dN2);

   int dim = 3;
   int num_dofs1 = dim;
   int num_dofs2 = 4*dim;
   int num_dofs = num_dofs1 + num_dofs2;
   dg2dx.SetSize(num_dofs,num_dofs); dg2dx = 0.0;

   Vector x_c(3);
   m_coords.MultTranspose(m_N,x_c);

   Vector gap_v(3); gap_v = 0.0;
   gap_v  = x_s;
   gap_v -= x_c;

   DenseMatrix m_dx(2,3);
   Mult(m_dN, m_coords, m_dx);

   DenseMatrix m_dx2(3,3); m_dx2 = 0.0;
   Mult(m_dN2,m_coords, m_dx2);
   double nnorm = 0.0;
   Vector normal(3); normal = 0.0;
   ComputeNormal(m_dN, m_coords, normal, nnorm);

   double gap = gap_v * normal; // gap function value, dot product between vectors

   DenseMatrix M(2,2); M = 0.0;
   MultABt(m_dx, m_dx, M);

   DenseMatrix f(2, num_dofs2); f = 0.0;

   for (int d=0; d<3; d++)
   {
      DenseMatrix Mtemp(2,2); Mtemp = 0.0;
      Mtemp(0,0) = m_dx2(0,d); Mtemp(0,1) = m_dx2(1,d);
      Mtemp(1,0) = m_dx2(1,d); Mtemp(1,1) = m_dx2(2,d);

      M.Add(-gap_v[d], Mtemp);

      Vector m_dxcol(2); m_dx.GetColumn(d, m_dxcol);
      DenseMatrix ftmp(2,4);
      outer(m_dxcol, m_N, ftmp);
      ftmp *= -1;
      ftmp.Add( gap_v[d], m_dN);   // 2*4

      for (int j=0; j<4; j++)
      {
         assert(d+3*j<num_dofs2);
         f(0,d+j*3) = ftmp(0,j);
         f(1,d+j*3) = ftmp(1,j);
      }
   }
   //fprintf('hess dxidxm\n');
   DenseMatrixInverse Minv(M);
   DenseMatrix dxidxm(2,num_dofs2); dxidxm = 0.0;
   Minv.Mult(f, dxidxm);
   //LinearSolve??
   //dxidxm = M\f;

   DenseMatrix nde2(2,2); nde2 = 0.0;
   DenseMatrix Nndx2(2,num_dofs2); Nndx2 = 0.0;

   for (int d=0; d<3; d++)
   {
      DenseMatrix ndetmp(2,2); ndetmp = 0.0;
      ndetmp(0,0) = normal(d)*m_dx2(0,d); ndetmp(0,1) = normal(d)*m_dx2(1,d);
      ndetmp(1,0) = normal(d)*m_dx2(1,d); ndetmp(1,1) = normal(d)*m_dx2(2,d);

      nde2 += ndetmp;

      for (int j=0; j<4; j++)
      {
         assert(d+3*j<num_dofs2);
         Nndx2(0,d+j*3) = normal[d]*m_dN(0,j);
         Nndx2(1,d+j*3) = normal[d]*m_dN(1,j);
      }

   }

   DenseMatrix Ndn(2,num_dofs2); Ndn = 0.0;
   Ndn += Nndx2;
   AddMult(nde2, dxidxm, Ndn);


   DenseMatrix M2(2,2); M2 = 0.0;
   MultABt(m_dx, m_dx, M2);
   DenseMatrixInverse M2inv(M2);
   DenseMatrix diag2(2,2); diag2(0,0) = 1.0; diag2(1,1) = 1.0;
   DenseMatrix m_con(2,2); m_con = 0.0;

   M2inv.Mult(diag2, m_con);

   DenseMatrix dg2dxm(num_dofs2, num_dofs2); dg2dxm = 0.0;

   DenseMatrix dg2dxm_tmp(num_dofs2,2); dg2dxm_tmp = 0.0;
   MultAtB(Ndn, m_con, dg2dxm_tmp);
   Mult(dg2dxm_tmp, Ndn, dg2dxm);
   dg2dxm *= gap;

   DenseMatrix dg2dxm_tmp2(num_dofs2,num_dofs2); dg2dxm_tmp2 = 0.0;
   MultAtB(Nndx2, dxidxm, dg2dxm_tmp2);
   dg2dxm.Add(-1.0, dg2dxm_tmp2);

   dg2dxm_tmp = 0.0;
   MultAtB(dxidxm, nde2, dg2dxm_tmp);

   AddMult_a(-1.0, dg2dxm_tmp, dxidxm, dg2dxm);

   dg2dxm_tmp2 = 0.0;
   MultAtB(dxidxm, Nndx2, dg2dxm_tmp2);
   dg2dxm.Add(-1.0, dg2dxm_tmp2);

   Vector v_dxidx2(4);
   m_coords.Mult(gap_v, v_dxidx2); // m_coords * gap_v;  // 4*3 * 3 = 4

   DenseMatrix K_dxidx2(2,2); K_dxidx2 = 0.0;

   Vector m_dN2row1(4); m_dN2.GetRow(0, m_dN2row1);
   Vector m_dN2row2(4); m_dN2.GetRow(1, m_dN2row2);
   Vector m_dN2row3(4); m_dN2.GetRow(2, m_dN2row3);
   K_dxidx2(0,0) = m_dN2row1 * v_dxidx2; // how would 4*1 * 1*4 be computed?
   K_dxidx2(0,1) = m_dN2row2 * v_dxidx2;
   K_dxidx2(1,0) = m_dN2row2 * v_dxidx2;
   K_dxidx2(1,1) = m_dN2row3 * v_dxidx2;

   DenseMatrix K_dxidx(2,2);
   K_dxidx -= M2;
   K_dxidx += K_dxidx2;

   Vector drdxs_r(6);
   drdxs_r[0] = m_dx(0,0); drdxs_r[1] = m_dx(0,1); drdxs_r[2] = m_dx(0,2);
   drdxs_r[3] = m_dx(1,0); drdxs_r[4] = m_dx(1,1); drdxs_r[5] = m_dx(1,2);

   DenseMatrix drdxs_K(6,6); drdxs_K = 0.;
   for (int i=0; i<3; i++)
   {
      drdxs_K(i,i) = K_dxidx(0,0);
      drdxs_K(i,3+i) = K_dxidx(0,1);
      drdxs_K(i+3,i) = K_dxidx(1,0);
      drdxs_K(i+3,i+3) = K_dxidx(1,1);
   }
   Vector dxidxs(6);

   DenseMatrixInverse drdxsK_inv(drdxs_K);
   drdxsK_inv.Mult(drdxs_r,dxidxs);
   dxidxs *= -1.0;
   //dxidxs = -drdxs_K\drdxs_r;

   DenseMatrix dxidxs_m(2,3); dxidxs_m = 0.0;
   dxidxs_m(0,0) = dxidxs[0]; dxidxs_m(0,1) = dxidxs[1]; dxidxs_m(0,2) = dxidxs[2];
   dxidxs_m(1,0) = dxidxs[3]; dxidxs_m(1,1) = dxidxs[4]; dxidxs_m(1,2) = dxidxs[5];

   DenseMatrix dtao1dxs(3,3); dtao1dxs = 0.0;
   DenseMatrix dtao2dxs(3,3); dtao2dxs = 0.0;

   Vector dxidxs_row1(3); dxidxs_row1 = 0.0; Vector dxidxs_row2(3);
   dxidxs_row2 = 0.0;
   Vector mdx2_row1(3); mdx2_row1 = 0.0; Vector mdx2_row2(3); mdx2_row2 = 0.0;
   Vector mdx2_row3(3); mdx2_row3 = 0.0;
   dxidxs_m.GetRow(0,dxidxs_row1);
   dxidxs_m.GetRow(1,dxidxs_row2);
   m_dx2.GetRow(0,mdx2_row1);
   m_dx2.GetRow(1,mdx2_row2);
   m_dx2.GetRow(2,mdx2_row3);

   DenseMatrix dtaotmp(3,3); dtaotmp = 0.0;
   outer(mdx2_row1, dxidxs_row1,dtaotmp);
   dtao1dxs += dtaotmp; dtaotmp = 0.0;
   outer(mdx2_row2, dxidxs_row1,dtaotmp);
   dtao1dxs += dtaotmp; dtaotmp = 0.0;

   outer(mdx2_row2, dxidxs_row2, dtaotmp);
   dtao2dxs += dtaotmp; dtaotmp = 0.0;
   outer(mdx2_row3, dxidxs_row2, dtaotmp);
   dtao2dxs += dtaotmp; dtaotmp = 0.0;

   DenseMatrix dtaodxs(3,3); dtaodxs = 0.0; //tao = tao1 cross tao2

   for (int d=0; d<3; d++)
   {
      Vector dtao1dxs_tmp(3); dtao1dxs_tmp = 0.0;
      dtao1dxs.GetColumn(d,dtao1dxs_tmp);
      Vector m_dxrow(3); m_dx.GetRow(1, m_dxrow);

      Vector dtaodxs_tmp(3); dtaodxs_tmp = 0.0;
      cross(dtao1dxs_tmp, m_dxrow, dtaodxs_tmp);

      Vector dtaodxs_tmp2(3); dtaodxs_tmp2 = 0.0;
      m_dx.GetRow(0, m_dxrow);
      dtao1dxs_tmp = 0.0;  // reuse the same vector for dtao2
      dtao2dxs.GetColumn(d,dtao1dxs_tmp);
      cross(m_dxrow, dtao1dxs_tmp, dtaodxs_tmp2);

      dtaodxs_tmp2 += dtaodxs_tmp;
      dtaodxs.SetCol(d, dtaodxs_tmp2);
   }

   DenseMatrix dndxs(3,3); dndxs = 0.0; dndxs += dtaodxs; dndxs *= 1.0/nnorm;
   DenseMatrix dndxs_tmp(3,3); dndxs_tmp = 0.0;
   outer(normal, normal, dndxs_tmp);
   AddMult_a(-1/nnorm, dndxs_tmp, dtaodxs, dndxs);

   DenseMatrix dgvdxs(3,3); dgvdxs = 0.0;
   MultAtB(m_dx, dxidxs_m, dgvdxs);
   dgvdxs *= -1;
   for (int d=0; d<3; d++)
   {
      dgvdxs(d,d) += 1.0;
   }
   //dxidxs: 2*3

   DenseMatrix dg2dxs(3,3); dg2dxs = 0.0;
   DenseMatrix dg2dxs_tmp(3,2); dg2dxs_tmp = 0.0;
   MultAtB(dxidxs_m, nde2, dg2dxs_tmp);
   AddMult_a(-1.0, dg2dxs_tmp, dxidxs_m, dg2dxs);
   DenseMatrix dg2dxs_tmp2(3,3); dg2dxs_tmp2 = 0.0;
   MultAtB(dgvdxs, dndxs, dg2dxs_tmp2);
   dg2dxs += dg2dxs_tmp2;
   dg2dxs_tmp2 = 0.0;
   MultAtB(dndxs, dndxs_tmp, dg2dxs_tmp2);
   AddMult(dg2dxs_tmp2, dgvdxs, dg2dxs);

   DenseMatrix Ne(3,12), Be(6,12), dBe(12,12);
   BasisVectorDerivs(xi, Ne, Be, dBe);

   DenseMatrix dtao1dxm(3,12); dtao1dxm.CopyRows(Be, 0, 2);
   DenseMatrix dtao2dxm(3,12); dtao2dxm.CopyRows(Be, 3, 5);

   Vector m_coords_v(12);
   for (int i=0; i<4; i++)
   {
      for (int j=0; j<3; j++)
      {
         m_coords_v[i*3+j] = m_coords(i,j);
      }
   }

   for (int i=0; i<2; i++)
   {
      Vector dxidxm_tmp(num_dofs2); dxidxm_tmp = 0.0;
      dxidxm.GetRow(i,dxidxm_tmp);

      DenseMatrix dBe_tmp(3,12);
      dBe_tmp.CopyRows(dBe,i*3,(i+1)*3-1);

      DenseMatrix dtaodxm_tmp(12,12); dtaodxm_tmp = 0.0;
      outer(m_coords_v, dxidxm_tmp, dtaodxm_tmp);
      AddMult(dBe_tmp, dtaodxm_tmp, dtao1dxm);

      //dtao1dxm += dBe(:,:,i)*reshape(m_coords(1:4,:)',12,1)*reshape(dxidxm(i,:),1,12); % 3*12
      dBe_tmp = 0.0;
      dBe_tmp.CopyRows(dBe,(i+2)*3,(i+3)*3-1);
      AddMult(dBe_tmp, dtaodxm_tmp, dtao2dxm);

   }

   DenseMatrix dtaodxm(3,12); dtaodxm = 0.0;//tao = tao1 cross tao2

   for (int d=0; d<12; d++)
   {
      Vector dtaodxm_tmp(3); dtaodxm_tmp = 0.0;
      Vector dtaodxm_tmp2(3); dtaodxm_tmp2 = 0.0;
      Vector tmp1(3); tmp1 = 0.0;  dtao1dxm.GetColumn(d,tmp1);
      Vector m_dxrow2(3); m_dx.GetRow(1, m_dxrow2);
      Vector m_dxrow1(3); m_dx.GetRow(0, m_dxrow1);
      Vector tmp2(3); tmp2 = 0.0;  dtao2dxm.GetColumn(d,tmp2);

      cross(tmp1, m_dxrow2, dtaodxm_tmp);
      cross(m_dxrow1,tmp2, dtaodxm_tmp2);
      dtaodxm_tmp += dtaodxm_tmp2;

      dtaodxm.SetCol(d, dtaodxm_tmp);
   }

   DenseMatrix dndxm(3,12); dndxm = 0.0;
   dndxm += dtaodxm;
   dndxm *= 1.0/nnorm;
   AddMult_a(-1/nnorm, dndxs_tmp, dtaodxm, dndxm); //dndxs_tmp = normal'*normal

   DenseMatrix dgvdxm(3,12); dgvdxm = 0.0;
   dgvdxm -= Ne;

   for (int i=0; i<2; i++)
   {
      Vector dxidxm_tmp(num_dofs2); dxidxm_tmp = 0.0;
      dxidxm.GetRow(i,dxidxm_tmp);

      DenseMatrix Be_tmp(3,12);
      Be_tmp.CopyRows(Be,i*3,(i+1)*3-1);

      DenseMatrix dgvdxm_tmp(12,12); dgvdxm_tmp = 0.0;
      outer(m_coords_v, dxidxm_tmp, dgvdxm_tmp);
      AddMult_a(-1.0, Be_tmp, dgvdxm_tmp, dgvdxm);

   }

   DenseMatrix dg2dxsxm(3,12); dg2dxsxm = 0.0;
   DenseMatrix dg2dxsxm_tmp(3,3); dg2dxsxm_tmp = 0.0;
   MultAtB(dgvdxs, dndxm, dg2dxsxm);

   MultAtB(dndxs, dndxs_tmp, dg2dxsxm_tmp);
   AddMult(dg2dxsxm_tmp, dgvdxm, dg2dxsxm); // += dndxs'*normal'*normal*dgvdxm;

   DenseMatrix dgvdxsxmn(3,12); dgvdxsxmn = 0.0;
   DenseMatrix dgvdxsxmn_tmp(3,2); dgvdxsxmn_tmp = 0.0;
   MultAtB(dxidxs_m, nde2, dgvdxsxmn_tmp);   //dxidxs_m: 2*3

   AddMult_a(-1.0, dgvdxsxmn_tmp, dxidxm, dgvdxsxmn);


   for (int i =0; i<2; i++)
   {
      DenseMatrix Be_tmp(3,12);
      Be_tmp.CopyRows(Be,i*3,(i+1)*3-1);

      Vector dxidxs_row(3); dxidxs_row = 0.0; dxidxs_m.GetRow(i,dxidxs_row);
      DenseMatrix dgvdxsxmn_tmp2(3,3); dgvdxsxmn_tmp2 = 0.0;
      outer(dxidxs_row, normal, dgvdxsxmn_tmp2);
      AddMult_a(-1.0, dgvdxsxmn_tmp2, Be_tmp, dgvdxsxmn);
   }

   dg2dxsxm += dgvdxsxmn;

   DenseMatrix dg2dxmxs(12,3); dg2dxmxs = 0.0;
   DenseMatrix dg2dxmxs_tmp(12,3); dg2dxmxs_tmp = 0.0;
   MultAtB(dgvdxm, dndxs, dg2dxmxs);
   MultAtB(dndxm, dndxs_tmp, dg2dxmxs_tmp);
   AddMult(dg2dxmxs_tmp, dgvdxs, dg2dxmxs);

   DenseMatrix dgvdxmxsn(12,3); dgvdxmxsn = 0.0;
   DenseMatrix dgvdxmxsn_tmp(12,2); dgvdxmxsn_tmp = 0.0;

   MultAtB(dxidxm, nde2, dgvdxmxsn_tmp);
   dgvdxmxsn_tmp *= -1.0;
   AddMult(dgvdxmxsn_tmp, dxidxs_m, dgvdxmxsn);

   for (int i =0; i<2; i++)
   {
      DenseMatrix Be_tmp(3,12);
      Be_tmp.CopyRows(Be,i*3,(i+1)*3-1);
      Be_tmp.Transpose(); // Be is now 12*3

      Vector dxidxs_row(3); dxidxs_row = 0.0; dxidxs_m.GetRow(i,dxidxs_row);
      DenseMatrix dgvdxmxsn_tmp2(3,3); dgvdxmxsn_tmp2 = 0.0;
      outer(normal, dxidxs_row, dgvdxmxsn_tmp2);
      AddMult_a(-1.0, Be_tmp, dgvdxmxsn_tmp2, dgvdxmxsn);

   }

   dg2dxmxs += dgvdxmxsn;

   dg2dx.CopyMN(dg2dxs, 0, 0);
   dg2dx.CopyMN(dg2dxm, 3, 3);
   dg2dx.CopyMN(dg2dxsxm, 0, 3);
   dg2dx.CopyMN(dg2dxmxs, 3, 0);

};



void NodeSegConPairs(const Vector x1, const Vector xi2,
                     const DenseMatrix coords2,
                     double& node_g, Vector& node_dg, DenseMatrix& node_dg2)
{
   double gap = 0.0;
   Vector normal(3); normal = 0.0;
   Vector dgdxm(12); dgdxm = 0.0;
   Vector dgdxs(3); dgdxs = 0.0;

   ComputeGapJacobian(x1, xi2, coords2, gap, normal, dgdxm, dgdxs);
   node_g = gap;

   node_dg.SetSize(12+3);
   for (int i=0; i<3; i++) { node_dg[i] = dgdxs[i]; }
   for (int i=0; i<12; i++) { node_dg[i+3] = dgdxm[i]; }

   DenseMatrix dg2dx(15,15); dg2dx = 0.0;
   DenseMatrix dgvdxmxsn(12,3); dgvdxmxsn = 0.0;
   ComputeGapHessian(x1, xi2, coords2, dg2dx);

   node_dg2.SetSize(15,15);
   node_dg2 = dg2dx;

   /*
   if(obj.space1.conns{e1}(i)==150)  % for debugging purpose

   v1 = 1:3;
   v2 = 1:12;
   %v1 = ones(1,3)
   %v2 = ones(1,12)
   v2 = reshape(v2,4,3);
   x1n1 = x1 + 0.01*v1;
   coords2n1 = coords2 + 0.001*v2;
      [xi2n1, gapv1, ~, ~] = SlaveToMaster(obj, coords2n1, x1n1);
      [gapn1, n1,dgdxmn1, dgdxsn1] = ComputeGapJacobian(obj, x1n1, xi2n1, coords2n1);
   x1n2 = x1 - 0.01*v1;
   coords2n2 = coords2 - 0.001*v2;
      [xi2n2, gapv2, ~, ~] = SlaveToMaster(obj, coords2n2, x1n2);
      [gapn2, n2,dgdxmn2, dgdxsn2] = ComputeGapJacobian(obj, x1n2, xi2n2, coords2n2);
   fprintf('fd\n');
      %gapv1-gapv2
   [dgdxsn1(:)',dgdxmn1(:)'] - [dgdxsn2(:)',dgdxmn2(:)']

   %dgdxsn1-dgdxsn2
   fprintf('code\n');
   v2n = v2';
   %dg2dx(1:3,1:3)*0.04*ones(3,1)
   temp = zeros(12,3);
   for i = 1:4
   temp1 = dg2dx(3+(i-1)*3+1:3+i*3,1:3);
   temp((i-1)*3+1:i*3,:) = temp1';
   end
   temp2 = zeros(3,12);
   for i = 1:4
   temp3 = dg2dx(1:3,3+(i-1)*3+1:3+i*3);
   temp2(:,(i-1)*3+1:i*3) = temp3';
   end
   %dg2dx
   %dg2dx(4:end,1:3) = temp;
   %dg2dx(1:3,4:end) = temp2;
   %dgvdxm * 0.002*v2n(:)
   (dg2dx*[0.02*v1(:)',0.002*v2n(:)']')'
   %dg2dx(4:end,1:3)
   end*/

};


// coordsm : (npoints*4, 3) use what class?
// m_conn: (npoints*4)
void Assemble_Contact(const int m, const int npoints, const int ndofs,
                      const Vector x_s,
                      const Vector xi, const DenseMatrix coordsm, const Array<int> s_conn,
                      const Array<int> m_conn, Vector& g, SparseMatrix& M,
                      std::vector<SparseMatrix>& dM)
{
   int ndim = 3;

   g.SetSize(m);
   g = 0.0;

   //SparseMatrix M(m, n);  // M needs to be the correct size

   //dM.resize(m); // needs to clear?

   double g_tmp = 0.;
   Vector dg(4*ndim+ndim);
   dg = 0.;
   DenseMatrix dg2(4*ndim+ndim,4*ndim+ndim);
   dg2 = 0.;

   for (int i=0; i<npoints; i++)
   {
      Vector x1(ndim);
      x1[0] = x_s[i*ndim];
      x1[1] = x_s[i*ndim+1];
      x1[2] = x_s[i*ndim+2];

      Vector xi2(ndim-1);
      xi2[0] = xi[i*(ndim-1)];
      xi2[1] = xi[i*(ndim-1)+1];

      DenseMatrix coords2(4,3);
      coords2.CopyRows(coordsm, i*4,(i+1)*4-1);

      //how to get coords2?
      dg = 0.0;
      dg2 = 0.;
      NodeSegConPairs(x1, xi2, coords2, g_tmp, dg, dg2);
      g[s_conn[i]] = g_tmp; // should be unique
      Array<int> m_conn_i(4);
      m_conn.GetSubArray(4*i, 4, m_conn_i);

      Array<int> node_conn(5);
      node_conn[0] = s_conn[i];
      for (int j=0; j<4; j++)
      {
         node_conn[j+1] = m_conn_i[j];
      }

      Array<int>  M_i_tmp(1);
      M_i_tmp[0] = s_conn[i];

      //j_idx = (node_conn-1)*obj.disp_field.num_components +repmat((1:obj.disp_field.num_components)', 1, length(node_conn{i}));
      Array<int> j_idx(5*ndim); j_idx = 0;
      for (int j=0; j< 5; j++)
      {
         for (int k=0; k<ndim; k++)
         {
            j_idx[j*ndim+k] = node_conn[j]*ndim+k;
         }
      }
      DenseMatrix M_v_tmp(1, ndim*(4+1));   // SetData now?
      M_v_tmp.SetRow(0, dg);

      M.AddSubMatrix(M_i_tmp, j_idx, M_v_tmp);

      Array<int> dM_i(ndim*(4+1));
      Array<int> dM_j(ndim*(4+1));

      for (int j=0; j< ndim*(4+1); j++)
      {
         dM_i[j] = j_idx[j];
         dM_j[j] = j_idx[j];
      }
      dM[s_conn[i]].AddSubMatrix(dM_i,dM_j, dg2);
      dM[s_conn[i]].Finalize();
      dM[s_conn[i]].Threshold(0.0);
      dM[s_conn[i]].SortColumnIndices();
   }
   M.Finalize();
   M.Threshold(0.0);
   M.SortColumnIndices();
};

