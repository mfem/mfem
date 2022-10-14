#ifndef IMPLICIT_PARAMETERIZATION
#define IMPLICIT_PARAMETERIZATION

#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>

#include "Function.hpp"

namespace ImplicitGeometry
{

class ImplicitParameterization
{
public:
   ImplicitParameterization(std::vector<double> params)
     : nParam_(params.size()), sdim_(3)
   {   
      x_.resize(sdim_);
      for (int i=0; i<sdim_; i++)
         {x_[i] = new Scalar(0.0);}

      params_.resize(nParam_);
      for (int i=0; i<nParam_; i++)
         {params_[i] = new Scalar(params[i]);}
   }

   int GetNParam() {return nParam_;}
   double GetParamValue(int i)
   {
      assert(0 <= i && i < nParam_);
      return params_[i]->Eval();
   }     
   void  SetParamValue(int i, double x)
   {
      assert(0 <= i && i < nParam_);
      params_[i]->SetValue(x);
   }
   void SetAllParams(std::vector<double> &params)
   {
      assert(params.size() == nParam_);
      for (int i=0; i<nParam_; i++)
         {this->SetParamValue(i, params[i]);}
   }
   double Eval(const std::vector<double> &x)
   {
      this->SetCoordinate(x);
      return this->FunctionOfInterest()->Eval();
   }
protected:
   virtual Function * FunctionOfInterest()=0;
   std::vector<Scalar*>      x_;
   std::vector<Scalar*> params_;
private:
   void    SetCoordinate(const std::vector<double> &x)
   {
      assert(x.size() == sdim_);
      for (int i=0; i<sdim_; i++)
         {x_[i]->SetValue(x[i]);}
   }
   int                    sdim_;
   int                  nParam_;
};

class Lattice2D : public ImplicitParameterization
{
public:
  Lattice2D(std::vector<double> params, double a=1.0)
    : ImplicitParameterization(params)
   {
      assert(params.size()==3);
      Function *lx = params_[0];
      Function *ly = params_[1];
      Function *a0 = params_[2];
      
      Function *A = new Scalar(a);
      double PI = 3.14159265358979323;
      Function *phi = new Scalar(-PI/2.0);
      Function *zero = new Scalar(0.0);
      Function *one = new Scalar(1.0);

      Function *sinX = new SineWave(x_[0], A, phi, lx);
      Function *sinY = new SineWave(x_[1], A, phi, ly);
      Function *lattice = new ROr(sinX, sinY, a0, one, one);

      Function *shift = new Scalar(0.5);
      Function *shift_x = new Subtraction(x_[0], shift);
      Function *quad = new Power(shift_x, 2.0);
      Function *offset = new Scalar((0.5-a/2.0)*(0.5-a/2.0));
      Function *skin = new Subtraction(quad, offset);

      f_ = new ROr(lattice, skin, zero, one, one);
   }
private: 
   Function *f_;
   Function* FunctionOfInterest() {return f_;}
};

}  //  Implicit Geometry namespace
#endif
