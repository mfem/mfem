#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ComplexCoefficient
{
protected:
   bool own_r=false;
   bool own_i=false;
   Coefficient * cr = nullptr;
   Coefficient * ci = nullptr;
public:
   ComplexCoefficient(){};
   ComplexCoefficient(Coefficient * cr_, Coefficient * ci_, 
   bool ownr_ = false, bool owni_ = false) 
   : cr(cr_), ci(ci_), own_r(own_r), own_i(owni_) {}
   void SetReal(Coefficient * cr_) { cr = cr_; }
   void SetImag(Coefficient * ci_) { ci = ci_; }
   Coefficient * real() { return cr; }
   Coefficient * imag() { return ci; }

   virtual ~ComplexCoefficient()
   {
      if (own_r) delete cr; 
      if (own_i) delete ci; 
   }
};

class RestrictedComplexCoefficient: public ComplexCoefficient
{
   RestrictedComplexCoefficient(ComplexCoefficient * A, Array<int> & attr)
   {
      if (A->real()) 
      {
         own_r = true;
         cr = new RestrictedCoefficient(*A->real(), attr);
      }
      if (A->imag()) 
      {
         own_i = true;
         ci = new RestrictedCoefficient(*A->imag(), attr);
      }
   }
};

class MatrixComplexCoefficient
{
protected:
   bool own_r = false;
   bool own_i = false;
   MatrixCoefficient * cr = nullptr;
   MatrixCoefficient * ci = nullptr;
public:
   MatrixComplexCoefficient(){};
   MatrixComplexCoefficient(MatrixCoefficient * cr_, MatrixCoefficient * ci_) : cr(cr_), ci(ci_) 
   { }
   void SetReal(MatrixCoefficient * cr_) { cr = cr_; }
   void SetImag(MatrixCoefficient * ci_) { ci = ci_; }
   MatrixCoefficient * real() {return cr;}
   MatrixCoefficient * imag() {return ci;}

   virtual ~MatrixComplexCoefficient()
   {
      if (own_r) delete cr; 
      if (own_i) delete ci; 
   }
};


class MatrixRestrictedComplexCoefficient: public MatrixComplexCoefficient
{
public:
   MatrixRestrictedComplexCoefficient(MatrixComplexCoefficient * A, Array<int> & attr)
   {
      if (A->real()) 
      {
         own_r = true;
         cr = new MatrixRestrictedCoefficient(*A->real(), attr);
      }
      if (A->imag()) 
      {
         own_i = true;
         ci = new MatrixRestrictedCoefficient(*A->imag(), attr);
      }
   }
};


class ProductComplexCoefficient: public ComplexCoefficient 
{
private:
   Coefficient * a = nullptr;
   Coefficient * b = nullptr;
   Coefficient * c = nullptr;
   Coefficient * d = nullptr;
   ComplexCoefficient * tmp = nullptr;
public:
   ProductComplexCoefficient(ComplexCoefficient * A, ComplexCoefficient * B);
   ProductComplexCoefficient(ComplexCoefficient * A, Coefficient * B) 
   {
      tmp = new ComplexCoefficient(B,nullptr);
      ProductComplexCoefficient(A,tmp);
   }
   ProductComplexCoefficient(Coefficient * A, ComplexCoefficient * B) 
   {
      tmp = new ComplexCoefficient(A,nullptr);
      ProductComplexCoefficient(tmp,B);
   }
   virtual ~ProductComplexCoefficient()
   {
      if(a) delete a;
      if(b) delete b;
      if(c) delete c;
      if(d) delete d;
      if (tmp) delete tmp;
      if (own_r) delete cr;
      if (own_i) delete ci;
   }
};

class ScalarMatrixProductComplexCoefficient: public MatrixComplexCoefficient 
{
private:
   MatrixCoefficient * a = nullptr;
   MatrixCoefficient * b = nullptr;
   MatrixCoefficient * c = nullptr;
   MatrixCoefficient * d = nullptr;
   ComplexCoefficient * tmp1 = nullptr;
   MatrixComplexCoefficient * tmp2 = nullptr;
   MatrixComplexCoefficient * tmp3 = nullptr;
   void Setup(ComplexCoefficient * A, MatrixComplexCoefficient * B);
public:
   ScalarMatrixProductComplexCoefficient(ComplexCoefficient * A, MatrixComplexCoefficient * B)
   {
      Setup(A,B);
   }
   
   ScalarMatrixProductComplexCoefficient(Coefficient * A, MatrixComplexCoefficient * B) 
   {
      tmp1 = new ComplexCoefficient(A,nullptr);
      Setup(tmp1,B);
   }
   ScalarMatrixProductComplexCoefficient(ComplexCoefficient * A, MatrixCoefficient * B) 
   {
      tmp2 = new MatrixComplexCoefficient(B,nullptr);
      Setup(A,tmp2);
   }

   virtual ~ScalarMatrixProductComplexCoefficient()
   {
      if(a) delete a;
      if(b) delete b;
      if(c) delete c;
      if(d) delete d;
      if (tmp1) delete tmp1;
      if (tmp2) delete tmp2;
      if (own_r) delete cr;
      if (own_i) delete ci;
   }
};

class MatrixMatrixProductComplexCoefficient: public MatrixComplexCoefficient 
{
private:
   MatrixCoefficient * a = nullptr;
   MatrixCoefficient * b = nullptr;
   MatrixCoefficient * c = nullptr;
   MatrixCoefficient * d = nullptr;
   MatrixComplexCoefficient * tmp = nullptr;
public:
   MatrixMatrixProductComplexCoefficient(MatrixComplexCoefficient * A, MatrixComplexCoefficient * B);
   MatrixMatrixProductComplexCoefficient(MatrixComplexCoefficient * A, MatrixCoefficient * B) 
   {
      tmp = new MatrixComplexCoefficient(B,nullptr);
      MatrixMatrixProductComplexCoefficient(A,tmp);
   }
   MatrixMatrixProductComplexCoefficient(MatrixCoefficient * A, MatrixComplexCoefficient * B) 
   {
      tmp = new MatrixComplexCoefficient(A,nullptr);
      MatrixMatrixProductComplexCoefficient(tmp,B);
   }
   virtual ~MatrixMatrixProductComplexCoefficient()
   {
      if(a) delete a;
      if(b) delete b;
      if(c) delete c;
      if(d) delete d;
      if (tmp) delete tmp;
      if (own_r) delete cr;
      if (own_i) delete ci;
   }
};

