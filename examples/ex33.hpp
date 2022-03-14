//              Implementation of the AAA algorithm
//
//   Here, we implement the triple-A algorithm [1] for the rational approximation
//   of complex-valued functions,
//
//          p(z)/q(z) ≈ f(z).
//
//   In this file, we always assume f(z) = z^{-α}. The triple-A algorithm
//   provides a robust, accurate approximation in rational barycentric from.
//   This representation must be transformed into a partial fraction
//   representation in order to be used to solve a spectral FPDE.
//
//   More specifically, we first expand the numerator in terms of the zeros of
//   the rational approximation,
//
//          p(z) ∝ Π_i (z - z_i),
//
//   and expand the denominator in terms of the poles of the rational
//   approximation,
//
//          q(z) ∝ Π_i (z - p_i).
//
//   We then use these zeros and poles to derive the partial fraction
//   expansion
//
//          f(z) ≈ p(z)/q(z) = Σ_i c_i / (z - p_i).
//
//
//   REFERENCE
//
//   [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA
//       algorithm for rational approximation. SIAM Journal on Scientific
//       Computing, 40(3), A1494-A1522.


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
   DenseMatrix *Ctemp = nullptr;
   DenseMatrix *A = nullptr;
   DenseMatrix *Am = nullptr;

   // auxiliary arrays and vectors
   Vector f_vec;
   Array<double> C_i;

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
      C_i.Append(C_tmp);
      int h_C = C_tmp.Size();
      int w_C = k+1;
      C.UseExternalData(C_i.GetData(),h_C,w_C);

      // This will need some cleanup
      // temporary copies to perform the scalings
      Ctemp = new DenseMatrix(C);

      f_vec.SetDataAndSize(f.GetData(),f.Size());
      Ctemp->InvLeftScaling(val);
      Ctemp->RightScaling(f_vec);

      A = new DenseMatrix(C.Height(), C.Width());
      Add(C,*Ctemp,-1.0,*A);
      A->LeftScaling(val);

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

      delete Ctemp;
      delete A;
      delete Am;

      if (verr.Normlinf() <= tol*val.Normlinf())
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
                                         double tol=1e-10,  int npoints = 1000,
                                         int max_order = 100)
{


#ifndef MFEM_USE_LAPACK
   mfem::out
         << "MFEM is compiled without LAPACK. Using precomputed PartialFractionApproximation"
         << std::endl;

   if (alpha == 0.33)
   {
      Array<double> temp_coeff({2002.55, 99.7691, 29.0575, 12.9842,
                                6.90263, 3.96688, 2.36013, 1.42565,
                                0.867623, 0.529436, 0.317975, 0.0891797});
      Array<double> temp_poles({-47928., -3451.45, -996.636, -388.58,
                                -168.228, -75.3162, -33.6235, -14.5403,
                                -5.84563, -1.9975, -0.434608, 0.});

      coeffs = temp_coeff;
      poles = temp_poles;
   }
   else if (alpha == 0.99)
   {
      Array<double> temp_coeff({0.0292386, 0.0143338, 0.0109015, 0.00976958,
                                0.00943417, 0.00948077, 0.00985799, 0.010889,
                                0.0138588, 0.0263289, 0.96953});
      Array<double> temp_poles({-10085.1, -1652.6, -524.342, -199.521,
                                -80.2328, -32.3926, -12.7148, -4.63855,
                                -1.39884, -0.221076, 0.});
      coeffs = temp_coeff;
      poles = temp_poles;
   }
   else
   {
      if (alpha != 0.5)
      {
         mfem::out << "Using default value of alpha = 0.5" << std::endl;
      }
      Array<double> temp_coeff({209.629, 24.2714, 9.24812, 4.93138,
                                3.02653, 1.98265, 1.34293, 0.931714,
                                0.664382, 0.492972, 0.39114, 0.177527});
      Array<double> temp_poles({-26466.7, -2673.76, -800.03, -312.646,
                                -134.551, -59.8651, -26.6607, -11.5603,
                                -4.67241, -1.59503, -0.332738, 0.});
      coeffs = temp_coeff;
      poles = temp_poles;
   }
   return;
#endif

   Vector x(npoints);
   Vector val(npoints);
   for (int i = 0; i<npoints; i++)
   {
      x(i) = (double)i+1.0;
      val(i) = pow(x(i),1.-alpha);
   }

   // Apply triple-A algorithm to f(x) = x^{1-a}
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

   // Introduce one extra pole at x=0, thus, delivering a
   // RA for f(x) = x^{-a}
   poles.Append(0.0);

   // Compute partial fraction approximation of f(x) = x^{-a}
   PartialFractionExpansion(scale, poles, zeros, coeffs);
}
