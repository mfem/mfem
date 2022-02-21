//              Implementation of the AAA algorithm
//
//   REFERENCE
//
//   Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA
//   algorithm for rational approximation. SIAM Journal on Scientific
//   Computing, 40(3), A1494-A1522.


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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
   DenseMatrix C;
   DenseMatrix *Cl = nullptr;
   DenseMatrix *Cr = nullptr;
   DenseMatrix *A = nullptr;
   DenseMatrix *Am = nullptr;

   // auxiliary arrays and vectors
   Vector f_vec;
   Array<double> C_i;

   // mean of the value vector
   Vector R(val.Size());
   double mean_val = val.Sum()/size;
   for (int i = 0; i<R.Size(); i++)
   {
      R(i) = mean_val;
   }

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
      // we might want to remove the inf
      for (int j = 0; j < size; j++)
      {
         C_tmp[j] = 1.0/(pt(j)-pt(idx));
      }
      C_i.Append(C_tmp);
      int h_C = C_tmp.Size();
      int w_C = k+1;
      C.UseExternalData(C_i.GetData(),h_C,w_C);

      // This will need some cleanup
      // temporary copies to perform the scalings
      Cl = new DenseMatrix(C);
      Cr = new DenseMatrix(C);

      f_vec.SetDataAndSize(f.GetData(),f.Size());
      Cl->LeftScaling(val);
      Cr->RightScaling(f_vec);

      A = new DenseMatrix(C.Height(), C.Width());
      Add(*Cl,*Cr,-1.0,*A);


      int h_Am = J.Size();
      int w_Am = A->Width();
      Am = new DenseMatrix(h_Am,w_Am);
      for (int i = 0; i<h_Am; i++)
      {
         int ii = J[i];
         for (int j = 0; j<w_Am; j++)
         {
            (*Am)(i,j) = (*A)(ii,j);
         }
      }

#ifdef MFEM_USE_LAPACK
      DenseMatrixSVD svd(*Am,false,true);
      svd.Eval(*Am);
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
      double err = verr.Normlinf();

      delete Cl;
      delete Cr;
      delete A;
      delete Am;

      if (err <= tol*val.Normlinf())
      {
         break;
      }
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
   double tmp1=0.0;
   double tmp2=0.0;
   for (int i = 0; i<m; i++)
   {
      tmp1 += w(i) * f(i);
      tmp2 += w(i);
   }
   scale = tmp1/tmp2;
}

void PartialFractionExpansion(double scale, Array<double> & poles,
                              Array<double> & zeros, Array<double> & coeffs)
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



// x^-a
void ComputePartialFractionApproximation(double alpha,
                                         Array<double> & coeffs, Array<double> & poles,
                                         double lmin = 1., double lmax = 1000.,
                                         double tol=1e-6,  int npoints = 1000,
                                         int max_order = 100)
{
   Vector x(npoints);
   Vector val(npoints);
   for (int i = 0; i<npoints; i++)
   {
      x(i) = (double)i+1.0;
      val(i) = pow(x(i),1.-alpha);
   }

   Array<double> z, f;
   Vector w;
   RationalApproximation_AAA(val,x,z,f,w,tol,max_order);

   Vector vecz, vecf;
   vecz.SetDataAndSize(z.GetData(), z.Size());
   vecf.SetDataAndSize(f.GetData(), f.Size());

   double scale;
   Array<double> zeros;
   ComputePolesAndZeros(vecz, vecf, w, poles, zeros, scale);

   poles.Append(0.0);
   PartialFractionExpansion(scale, poles, zeros, coeffs);
}
