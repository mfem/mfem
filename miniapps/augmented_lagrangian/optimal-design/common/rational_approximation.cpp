
#include "rational_approximation.hpp"

void RationalApproximation_AAA(const Vector &val, const Vector &pt,
                               Array<double> &z, Array<double> &f, Vector &w,
                               double tol, int max_order)
{

   // number of sample points
   int size = val.Size();
   MFEM_VERIFY(pt.Size() == size, "size mismatch");

   // Initializations
   Array<int> J(size);
   for (int i = 0; i < size; i++) { J[i] = i; }
   z.SetSize(0);
   f.SetSize(0);

   DenseMatrix C, Ctemp, A, Am;
   // auxiliary arrays and vectors
   Vector f_vec;
   Array<double> c_i;

   // mean of the value vector
   Vector R(val.Size());
   double mean_val = val.Sum()/size;

   for (int i = 0; i<R.Size(); i++) { R(i) = mean_val; }

   for (int k = 0; k < max_order; k++)
   {
      // select next support point
      int idx = 0;
      double tmp_max = 0;
      for (int j = 0; j < size; j++)
      {
         double tmp = abs(val(j)-R(j));
         if (tmp > tmp_max)
         {
            tmp_max = tmp;
            idx = j;
         }
      }

      // Append support points and data values
      z.Append(pt(idx));
      f.Append(val(idx));

      // Update index vector
      J.DeleteFirst(idx);

      // next column in Cauchy matrix
      Array<double> C_tmp(size);
      for (int j = 0; j < size; j++)
      {
         C_tmp[j] = 1.0/(pt(j)-pt(idx));
      }
      c_i.Append(C_tmp);
      int h_C = C_tmp.Size();
      int w_C = k+1;
      C.UseExternalData(c_i.GetData(),h_C,w_C);

      Ctemp = C;

      f_vec.SetDataAndSize(f.GetData(),f.Size());
      Ctemp.InvLeftScaling(val);
      Ctemp.RightScaling(f_vec);

      A.SetSize(C.Height(), C.Width());
      Add(C,Ctemp,-1.0,A);
      A.LeftScaling(val);

      int h_Am = J.Size();
      int w_Am = A.Width();
      Am.SetSize(h_Am,w_Am);
      for (int i = 0; i<h_Am; i++)
      {
         int ii = J[i];
         for (int j = 0; j<w_Am; j++)
         {
            Am(i,j) = A(ii,j);
         }
      }

#ifdef MFEM_USE_LAPACK
      DenseMatrixSVD svd(Am,false,true);
      svd.Eval(Am);
      DenseMatrix &v = svd.RightSingularvectors();
      v.GetRow(k,w);
#else
      mfem_error("Compiled without LAPACK");
#endif

      // N = C*(w.*f); D = C*w; % numerator and denominator
      Vector aux(w);
      aux *= f_vec;
      Vector N(C.Height()); // Numerator
      C.Mult(aux,N);
      Vector D(C.Height()); // Denominator
      C.Mult(w,D);

      R = val;
      for (int i = 0; i<J.Size(); i++)
      {
         int ii = J[i];
         R(ii) = N(ii)/D(ii);
      }

      Vector verr(val);
      verr-=R;

      if (verr.Normlinf() <= tol*val.Normlinf()) { break; }
   }
}


void ComputePolesAndZeros(const Vector &z, const Vector &f, const Vector &w,
                          Array<double> & poles, Array<double> & zeros, double &scale)
{
   // Initialization
   poles.SetSize(0);
   zeros.SetSize(0);

   // Compute the poles
   int m = w.Size();
   DenseMatrix B(m+1); B = 0.;
   DenseMatrix E(m+1); E = 0.;
   for (int i = 1; i<=m; i++)
   {
      B(i,i) = 1.;
      E(0,i) = w(i-1);
      E(i,0) = 1.;
      E(i,i) = z(i-1);
   }

#ifdef MFEM_USE_LAPACK
   DenseMatrixGeneralizedEigensystem eig1(E,B);
   eig1.Eval();
   Vector & evalues = eig1.EigenvaluesRealPart();
   for (int i = 0; i<evalues.Size(); i++)
   {
      if (IsFinite(evalues(i)))
      {
         poles.Append(evalues(i));
      }
   }
#else
   mfem_error("Compiled without LAPACK");
#endif
   // compute the zeros
   B = 0.;
   E = 0.;
   for (int i = 1; i<=m; i++)
   {
      B(i,i) = 1.;
      E(0,i) = w(i-1) * f(i-1);
      E(i,0) = 1.;
      E(i,i) = z(i-1);
   }

#ifdef MFEM_USE_LAPACK
   DenseMatrixGeneralizedEigensystem eig2(E,B);
   eig2.Eval();
   evalues = eig2.EigenvaluesRealPart();
   for (int i = 0; i<evalues.Size(); i++)
   {
      if (IsFinite(evalues(i)))
      {
         zeros.Append(evalues(i));
      }
   }
#else
   mfem_error("Compiled without LAPACK");
#endif
   for (int i = 0; i<f.Size(); i++)
   {
      if (abs(f(i))<1e-12 || !IsFinite(f(i))) continue;

      double num = 1.0, denom = 1.0;
      for (int j = 0; j<zeros.Size(); j++)
      {
         num *= z(i)-zeros[j];
      }
      for (int j = 0; j<poles.Size(); j++)
      {
         denom *= z(i)-poles[j];
      }
      scale = f(i)/(num/denom);
      break;
   }

   // scale = w*f / w.Sum();
}

void PartialFractionExpansion(double scale, const Array<double> & poles,
                              const Array<double> & zeros, Array<double> & coeffs)
{
   int psize = poles.Size();
   int zsize = zeros.Size();
   coeffs.SetSize(psize);
   coeffs = scale;

   for (int i=0; i<psize; i++)
   {
      double tmp_numer=1.0;
      for (int j=0; j<zsize; j++)
      {
         tmp_numer *= poles[i]-zeros[j];
      }

      double tmp_denom=1.0;
      for (int k=0; k<psize; k++)
      {
         if (k != i) { tmp_denom *= poles[i]-poles[k]; }
      }
      coeffs[i] *= tmp_numer / tmp_denom;
   }
}

void ComputePartialFractionApproximation(double alpha, double beta,
                                         Array<double> & coeffs, Array<double> & poles,
                                         double lmax, double tol, int npoints, int max_order)
{
   MFEM_VERIFY(alpha <= 1., "alpha must be less than 1");
   MFEM_VERIFY(alpha > 0., "alpha must be greater than 0");
   MFEM_VERIFY(npoints > 2, "npoints must be greater than 2");
   MFEM_VERIFY(lmax > 0,  "lmin must be greater than 0");
   MFEM_VERIFY(tol > 0,  "tol must be greater than 0");

   bool print_warning = true;
#ifdef MFEM_USE_MPI
   if ((Mpi::IsInitialized() && !Mpi::Root())) { print_warning = false; }
#endif

   Vector x(npoints);
   Vector val(npoints);
   double dx = lmax / (double)(npoints-1);
   for (int i = 0; i<npoints; i++)
   {
      x(i) = dx * (double)i;
      // val(i) = 1.0/(pow(x(i),alpha) + beta);
      val(i) = pow(x(i),alpha) + beta;
   }
   // Apply triple-A algorithm to f(x) = x^{a} + b
   Array<double> z, f;
   Vector w;
   RationalApproximation_AAA(val,x,z,f,w,tol,max_order);

   Vector vecz, vecf;
   vecz.SetDataAndSize(z.GetData(), z.Size());
   vecf.SetDataAndSize(f.GetData(), f.Size());

   // Compute poles and zeros for RA of f(x) = x^{1-a}
   double scale;
   Array<double> zeros;
   ComputePolesAndZeros(vecz, vecf, w, poles, zeros, scale);

   // Compute partial fraction approximation of f(x) = (x^{a} + b)^{-1}
   
   // swap poles with zeros
   Array<double> & temp_poles = zeros;
   Array<double> & temp_zeros = poles;
   scale = 1.0/scale;
   
   PartialFractionExpansion(scale, temp_poles, temp_zeros, coeffs);
   poles = zeros;

}
