//                               Implementation of the AAA algorithm
//
//
//
//
//
//
//
//
//
//
//   REFERENCE
//
//   Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA
//   algorithm for rational approximation. SIAM Journal on Scientific
//   Computing, 40(3), A1494-A1522.


extern "C" void
dggev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *B,
       int *ldb, double *alphar, double *alphai, double *beta, double *vl,
       int * ldvl, double * vr, int * ldvr, double * work, int * lwork, int* info);



#include "mfem.hpp"

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void RationalApproximation_AAA(const Vector &val,
                               const Vector &pt,           // inputs
                               Array<double> &z, Array<double> &f, Vector &w, // outputs
                               double tol = 1.0e-8, int max_order = 100);

void ComputePolesAndZeros(const Vector &z, const Vector &f, const Vector &w,
                          Vector & poles, Vector & zeros);



int main(int argc, char *argv[])
{
   int size = 10;
   Vector x(size);
   Vector val(size);
   for (int i = 0; i<size; i++)
   {
      x(i) = (double)i+0.01;
      val(i) = 1./pow(x(i),0.33);
   }

   Array<double> z, f;
   Vector w;
   RationalApproximation_AAA(val,x,z,f,w,1e-7,100);

   mfem::out << "x = " ; x.Print(cout,size);
   mfem::out << "val = " ; val.Print(cout,size);
   mfem::out << endl;
   mfem::out << "z = " ; z.Print(cout,z.Size());
   mfem::out << "f = " ; f.Print(cout,f.Size());
   mfem::out << "w = " ; w.Print(cout,w.Size());


   Vector vecz, vecf;
   vecz.SetDataAndSize(z.GetData(), z.Size());
   vecf.SetDataAndSize(f.GetData(), f.Size());

   Vector poles, zeros;
   ComputePolesAndZeros(vecz, vecf, w, poles, zeros);

}


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

   // mfem::out << "Val mean = " << mean_val << endl;
   // Array<double> errors;
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
      // mfem::out << "tmp_max = " << tmp_max << endl;
      // mfem::out << "idx = " << idx << endl;

      // Append support points and data values
      z.Append(pt(idx));
      f.Append(val(idx));

      // mfem::out << "z = "; z.Print();
      // mfem::out << "f = "; f.Print();

      // Update index vector
      J.DeleteFirst(idx);
      // mfem::out << "J = "; J.Print(cout, J.Size());

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
      // mfem::out << "C = " << endl;
      // C.PrintMatlab();

      // This will need some cleanup
      // temporary copies to perform the scalings
      Cl = new DenseMatrix(C);
      Cr = new DenseMatrix(C);

      f_vec.SetDataAndSize(f.GetData(),f.Size());
      Cl->LeftScaling(val);
      Cr->RightScaling(f_vec);

      A = new DenseMatrix(C.Height(), C.Width());
      Add(*Cl,*Cr,-1.0,*A);

      // remove select only the J indexes of the columns of A

      // mfem::out << "A = " << endl;
      // A->PrintMatlab();

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


      // mfem::out << "Am = " << endl;
      // Am->PrintMatlab();

      // SVD to get the eigenvectors of Am

      DenseMatrix AtA(Am->Width());
      MultAtB(*Am,*Am,AtA);

      DenseMatrixEigensystem eig(AtA);
      eig.Eval();
      DenseMatrix & v = eig.Eigenvectors();
      Vector & svalues = eig.Eigenvalues();

      for (int i = 0; i<svalues.Size(); i++)
      {
         svalues(i) = std::sqrt(svalues(i));
      }

      // The columns of v (right-singular vectors) of A are eigenvectors of AtA
      // Singular values of A are the square roots of eigenvalues of AtA
      // mfem::out << "svalues = " << endl;
      // svalues.Print();
      // mfem::out << "singular vectors = " << endl;
      // v.PrintMatlab();

      v.GetColumn(0,w);

      // mfem::out << "w = " ; w.Print();

      // N = C*(w.*f); D = C*w; % numerator and denominator
      Vector aux(w);
      aux *= f_vec;
      Vector N(C.Height()); // Numerator
      C.Mult(aux,N);
      Vector D(C.Height()); // Denominator
      C.Mult(w,D);

      // mfem::out << "Numerator = "; N.Print();
      // mfem::out << "Denominator = "; D.Print();

      R = val;
      for (int i = 0; i<J.Size(); i++)
      {
         int ii = J[i];
         R(ii) = N(ii)/D(ii);
      }

      // mfem::out << "R = "; R.Print();

      Vector verr(val);
      verr-=R;
      double err = verr.Normlinf();
      // errors.Append(err);


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
                          Vector & poles, Vector & zeros)
{
   //    m = length(w); B = eye(m+1); B(1,1) = 0;
   //    E = [0 w.'; ones(m,1) diag(z)];
   //    pol = eig(E,B)

   // compute the poles

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

   Vector alphar(m+1);
   Vector alphai(m+1);
   Vector beta(m+1);

   int lwork = 8*(m+1);
   Vector work(lwork);
   int info;

   char jobvl  = 'N';
   char jobvr  = 'N';


   int N = m+1;
   int M = 1;

   dggev_(&jobvl,&jobvr,&N,E.GetData(),&N,B.GetData(),&N,alphar.GetData(),
          alphai.GetData(), beta.GetData(), nullptr, &M, nullptr, &M,
          work.GetData(), &lwork, &info);

   mfem::out << "info = " << info << endl;

   for (int i = 0; i<=m; i++ )
   {
      if (beta(i) == 0.)
      {
         mfem::out << "poles("<<i<<") = inf" << endl;
      }
      else
      {
         mfem::out << "poles("<<i<<") = " << alphar(i)/beta(i) <<
                   " + " << alphai(i)/beta(i) << " i " << endl;
      }
   }


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


   dggev_(&jobvl,&jobvr,&N,E.GetData(),&N,B.GetData(),&N,alphar.GetData(),
          alphai.GetData(), beta.GetData(), nullptr, &M, nullptr, &M,
          work.GetData(), &lwork, &info);

   mfem::out << "info = " << info << endl;

   for (int i = 0; i<=m; i++ )
   {
      if (beta(i) == 0.)
      {
         mfem::out << "zeros("<<i<<") = inf" << endl;
      }
      else
      {
         mfem::out << "zeros("<<i<<") = " << alphar(i)/beta(i) <<
                   " + " << alphai(i)/beta(i) << " i " << endl;
      }
   }


}